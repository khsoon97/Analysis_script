"""
Utilities for saving the V9 CODEX frozen ensemble and performing external inference.

V8 fits RF/XGB/ENet/MLP after applying an iteration-specific StandardScaler,
then averages probabilities with equal weights over the selected models and iterations.
Without changing these calculation or selection rules, this module saves the
objects and metadata needed for reuse immediately after training.

External data must already be a matrix in the CODEX marker feature space.
Creating scRNA-seq gene proxies is the responsibility of dataset-specific preprocessing;
this module does not arbitrarily fill missing features with zeros.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from tqdm.auto import tqdm


TRANSFER_FORMAT_VERSION = "1.0"
PIPELINE_VERSION = "V9"
PIPELINE_MINOR_VERSION = "mv1"
TRANSFER_MODEL_KEYS = ("RF", "XGB", "ENet", "MLP")


def _json_value(value: Any) -> Any:
    """Convert NumPy/Pandas values to JSON-serializable primitive types."""
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [_json_value(v) for v in list(value)]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def _write_json_atomic(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    """Save atomically to avoid leaving a corrupt manifest after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def _read_json(path: os.PathLike[str] | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_stage_name(value: str) -> str:
    """Convert a lineage name to a safe directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("_") or "unnamed"


def transfer_stage_dir(
    transfer_root: os.PathLike[str] | str,
    stage: str,
    lineage: str | None = None,
) -> Path:
    root = Path(transfer_root)
    if stage == "broad":
        return root / "broad"
    if stage == "subtype":
        if not lineage:
            raise ValueError("The subtype stage requires a lineage.")
        return root / "subtype" / safe_stage_name(lineage)
    raise ValueError(f"Unsupported stage: {stage}")


def prepare_transfer_stage(
    transfer_root: os.PathLike[str] | str,
    *,
    stage: str,
    feature_names: Sequence[str],
    classes: Sequence[str],
    expected_iterations: int,
    final_combo: str | None = None,
    lineage: str | None = None,
    overwrite: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Prepare a new frozen-model stage and create its initial manifest."""
    stage_dir = transfer_stage_dir(transfer_root, stage, lineage)
    manifest_path = stage_dir / "manifest.json"

    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"A V9 transfer package already exists: {stage_dir}. "
            "Set V9_TRANSFER_OVERWRITE_EXISTING=True to replace it with newly trained results."
        )
    if stage_dir.exists() and overwrite:
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": TRANSFER_FORMAT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_minor_version": PIPELINE_MINOR_VERSION,
        "stage": stage,
        "lineage": lineage,
        "selected_combo": final_combo,
        "selected_iterations": None,
        "model_keys": list(TRANSFER_MODEL_KEYS),
        "expected_iterations": int(expected_iterations),
        "feature_names": [str(x) for x in feature_names],
        "classes": [str(x) for x in classes],
        "preprocessing": "iteration-specific sklearn.preprocessing.StandardScaler",
        "probability_rule": (
            "Collect predict_proba for each iteration of the user-selected model combination, then "
            "take the arithmetic mean with equal weights across models and iterations"
        ),
        "available_iterations": {},
        "all_model_complete_iterations": [],
        "status": "training",
        "metadata": dict(metadata or {}),
    }
    _write_json_atomic(manifest_path, manifest)
    return stage_dir


def iteration_dir(stage_dir: os.PathLike[str] | str, iteration: int) -> Path:
    return Path(stage_dir) / f"iter_{int(iteration):03d}"


def save_iteration_scaler(
    stage_dir: os.PathLike[str] | str,
    iteration: int,
    scaler: Any,
    *,
    random_seed: int,
    train_n: int,
    validation_n: int,
    train_patient_n: int | None = None,
    validation_patient_n: int | None = None,
) -> Path:
    """Save the iteration-specific scaler and split provenance."""
    out_dir = iteration_dir(stage_dir, iteration)
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = out_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    _write_json_atomic(
        out_dir / "iteration.json",
        {
            "iteration": int(iteration),
            "random_seed": int(random_seed),
            "train_n": int(train_n),
            "validation_n": int(validation_n),
            "train_patient_n": train_patient_n,
            "validation_patient_n": validation_patient_n,
            "files": {"scaler": scaler_path.name},
            "saved_models": [],
        },
    )
    return scaler_path


def _register_iteration_file(
    stage_dir: os.PathLike[str] | str,
    iteration: int,
    model_key: str,
    model_path: os.PathLike[str] | str,
) -> None:
    meta_path = iteration_dir(stage_dir, iteration) / "iteration.json"
    meta = _read_json(meta_path, default={})
    files = dict(meta.get("files", {}))
    files[str(model_key).lower()] = Path(model_path).name
    saved_models = list(meta.get("saved_models", []))
    if model_key not in saved_models:
        saved_models.append(model_key)
    meta["files"] = files
    meta["saved_models"] = saved_models
    _write_json_atomic(meta_path, meta)


def save_xgb_iteration(
    stage_dir: os.PathLike[str] | str,
    iteration: int,
    model: Any,
) -> Path:
    """Save XGBoost in native JSON, which is more portable than Python pickle."""
    out_path = iteration_dir(stage_dir, iteration) / "xgb_model.json"
    model.save_model(str(out_path))
    _register_iteration_file(stage_dir, iteration, "XGB", out_path)
    return out_path


def save_joblib_model_iteration(
    stage_dir: os.PathLike[str] | str,
    iteration: int,
    model_key: str,
    model: Any,
) -> Path:
    """Save an RF or ENet model as its original library object using joblib."""
    model_key = str(model_key).upper()
    if model_key not in {"RF", "ENET"}:
        raise ValueError(f"Not a joblib save target: {model_key}")
    filename = "rf_model.joblib" if model_key == "RF" else "enet_model.joblib"
    out_path = iteration_dir(stage_dir, iteration) / filename
    joblib.dump(model, out_path)
    _register_iteration_file(
        stage_dir,
        iteration,
        "RF" if model_key == "RF" else "ENet",
        out_path,
    )
    return out_path


def mlp_iteration_path(stage_dir: os.PathLike[str] | str, iteration: int) -> Path:
    """Return the state-package path written directly by the MLP subprocess."""
    out_dir = iteration_dir(stage_dir, iteration)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "mlp_model.pt"


def register_mlp_iteration(
    stage_dir: os.PathLike[str] | str,
    iteration: int,
    model_path: os.PathLike[str] | str,
) -> None:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MLP model package was not created: {model_path}")
    _register_iteration_file(stage_dir, iteration, "MLP", model_path)


def finalize_transfer_stage(
    stage_dir: os.PathLike[str] | str,
    *,
    selected_iterations: int | None = None,
    selected_combo: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check saved iterations and mark the stage manifest as complete."""
    stage_dir = Path(stage_dir)
    manifest_path = stage_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"Transfer manifest not found: {manifest_path}")

    available = {key: [] for key in TRANSFER_MODEL_KEYS}
    scaler_iterations: list[int] = []
    for out_dir in sorted(stage_dir.glob("iter_*")):
        try:
            iteration = int(out_dir.name.split("_")[-1])
        except ValueError:
            continue
        if (out_dir / "scaler.joblib").exists():
            scaler_iterations.append(iteration)
        if (out_dir / "xgb_model.json").exists():
            available["XGB"].append(iteration)
        if (out_dir / "rf_model.joblib").exists():
            available["RF"].append(iteration)
        if (out_dir / "enet_model.joblib").exists():
            available["ENet"].append(iteration)
        if (out_dir / "mlp_model.pt").exists():
            available["MLP"].append(iteration)

    complete_set = set(scaler_iterations)
    for model_key in TRANSFER_MODEL_KEYS:
        complete_set &= set(available[model_key])
    complete = sorted(complete_set)
    manifest["available_iterations"] = available
    manifest["scaler_iterations"] = sorted(scaler_iterations)
    manifest["all_model_complete_iterations"] = complete
    expected_n = int(manifest.get("expected_iterations", 0))
    manifest["status"] = (
        "complete" if expected_n > 0 and len(complete) >= expected_n else "partial"
    )
    if selected_combo is not None:
        manifest["selected_combo"] = selected_combo
    if selected_iterations is not None:
        manifest["selected_iterations"] = int(selected_iterations)
    if extra_metadata:
        merged = dict(manifest.get("metadata", {}))
        merged.update(dict(extra_metadata))
        manifest["metadata"] = merged
    _write_json_atomic(manifest_path, manifest)
    build_transfer_package_index(stage_dir.parents[0] if manifest["stage"] == "broad" else stage_dir.parents[1])
    return manifest


def set_transfer_selection(
    stage_dir: os.PathLike[str] | str,
    combo: str,
    selected_iterations: int,
) -> dict[str, Any]:
    """Record the user-selected final combo/iterations in the manifest, as in V8."""
    stage_dir = Path(stage_dir)
    manifest_path = stage_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"Transfer manifest not found: {manifest_path}")
    model_keys = [part.strip() for part in str(combo).split("+") if part.strip()]
    invalid = [key for key in model_keys if key not in TRANSFER_MODEL_KEYS]
    if not model_keys or invalid:
        raise ValueError(f"Unsupported model combination: {combo}; invalid={invalid}")

    available = manifest.get("available_iterations", {})
    scaler_iterations = set(manifest.get("scaler_iterations", []))
    selected_model_iterations = {
        model_key: [
            int(i) for i in available.get(model_key, [])
            if int(i) in scaler_iterations
        ][: int(selected_iterations)]
        for model_key in model_keys
    }
    manifest["selected_combo"] = "+".join(model_keys)
    manifest["selected_iterations"] = int(selected_iterations)
    manifest["selected_model_iterations"] = selected_model_iterations
    _write_json_atomic(manifest_path, manifest)
    build_transfer_package_index(
        stage_dir.parents[0] if manifest["stage"] == "broad" else stage_dir.parents[1]
    )
    return manifest


def build_transfer_package_index(transfer_root: os.PathLike[str] | str) -> dict[str, Any]:
    """Combine broad/subtype stage manifests into one package index."""
    root = Path(transfer_root)
    stage_manifests: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("**/manifest.json")):
        if manifest_path == root / "package_manifest.json":
            continue
        payload = _read_json(manifest_path)
        if not payload:
            continue
        payload = dict(payload)
        payload["relative_manifest"] = str(manifest_path.relative_to(root))
        stage_manifests.append(payload)

    index = {
        "format_version": TRANSFER_FORMAT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_minor_version": PIPELINE_MINOR_VERSION,
        "purpose": "V8 model-selection-compatible frozen CODEX ensemble transfer package",
        "stages": stage_manifests,
    }
    _write_json_atomic(root / "package_manifest.json", index)
    return index


class SimpleMLP(nn.Module):
    """Inference network identical to V9 `_mlp_subprocess_mv1.py`."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def align_feature_matrix(
    X: pd.DataFrame | np.ndarray,
    expected_feature_names: Sequence[str],
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Strictly align an external matrix to the training feature order."""
    expected = [str(x) for x in expected_feature_names]
    if isinstance(X, pd.DataFrame):
        duplicated = X.columns[X.columns.duplicated()].astype(str).tolist()
        if duplicated:
            raise ValueError(f"Duplicate feature columns: {duplicated}")
        missing = [name for name in expected if name not in X.columns]
        if missing:
            raise KeyError(f"Features required for CODEX transfer are missing: {missing}")
        return np.ascontiguousarray(X.loc[:, expected].to_numpy(dtype=np.float32))

    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(f"X must be a two-dimensional matrix: shape={array.shape}")
    if feature_names is None:
        if array.shape[1] != len(expected):
            raise ValueError(
                f"Column count mismatch for a matrix supplied without feature_names: "
                f"received={array.shape[1]}, expected={len(expected)}"
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    provided = [str(x) for x in feature_names]
    lookup = {name: idx for idx, name in enumerate(provided)}
    missing = [name for name in expected if name not in lookup]
    if missing:
        raise KeyError(f"Features required for CODEX transfer are missing: {missing}")
    return np.ascontiguousarray(array[:, [lookup[name] for name in expected]], dtype=np.float32)


def _align_probability_columns(probs: np.ndarray, model_classes: Iterable[int] | None, n_classes: int) -> np.ndarray:
    probs = np.asarray(probs)
    if probs.ndim == 1 and n_classes == 2:
        probs = np.column_stack([1.0 - probs, probs])
    if probs.shape[1] == n_classes:
        return probs.astype(np.float32, copy=False)
    aligned = np.zeros((probs.shape[0], n_classes), dtype=np.float32)
    if model_classes is None:
        aligned[:, : min(probs.shape[1], n_classes)] = probs[:, :n_classes]
        return aligned
    for src_idx, cls_idx in enumerate(np.asarray(list(model_classes), dtype=int)):
        if src_idx < probs.shape[1] and 0 <= cls_idx < n_classes:
            aligned[:, cls_idx] = probs[:, src_idx]
    return aligned


def _load_torch_package(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _predict_mlp_once(
    model_path: Path,
    X_scaled: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    package = None
    model = None
    batch = None
    probs: list[np.ndarray] = []
    try:
        package = _load_torch_package(model_path, device)
        model = SimpleMLP(
            input_dim=int(package["input_dim"]),
            n_classes=int(package["n_classes"]),
            hidden_dims=package.get("hidden_dims", [256, 128]),
            dropout_rate=float(package.get("dropout_rate", 0.5)),
        ).to(device)
        model.load_state_dict(package["state_dict"])
        model.eval()

        with torch.inference_mode():
            for start in range(0, X_scaled.shape[0], batch_size):
                batch = torch.as_tensor(
                    X_scaled[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                probs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)
    finally:
        batch = None
        model = None
        package = None
        probs.clear()
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _is_cuda_runtime_failure(exc: RuntimeError) -> bool:
    message = str(exc).upper()
    return any(token in message for token in ("CUDA", "CUBLAS", "CUDNN"))


def _predict_mlp(
    model_path: Path,
    X_scaled: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, torch.device]:
    """Apply the saved MLP and retry on CPU with the same weights if CUDA fails."""
    try:
        return (
            _predict_mlp_once(
                model_path,
                X_scaled,
                device=device,
                batch_size=batch_size,
            ),
            device,
        )
    except RuntimeError as exc:
        if device.type != "cuda" or not _is_cuda_runtime_failure(exc):
            raise
        warnings.warn(
            "CUDA inference failed for the saved frozen MLP; retrying the iteration from the "
            f"beginning on CPU using the same scaler and model weights: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        cpu_device = torch.device("cpu")
        return (
            _predict_mlp_once(
                model_path,
                X_scaled,
                device=cpu_device,
                batch_size=batch_size,
            ),
            cpu_device,
        )


def predict_frozen_stage(
    X: pd.DataFrame | np.ndarray,
    stage_dir: os.PathLike[str] | str,
    *,
    feature_names: Sequence[str] | None = None,
    n_iterations: int | None = None,
    combo: str | None = None,
    batch_size: int = 8192,
    device: str = "auto",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Apply the saved ensemble with the selected model combination without retraining."""
    stage_dir = Path(stage_dir)
    manifest = _read_json(stage_dir / "manifest.json")
    if not manifest:
        raise FileNotFoundError(f"Transfer manifest not found: {stage_dir}")

    expected_features = manifest["feature_names"]
    classes = np.asarray(manifest["classes"], dtype=object)
    X_aligned = align_feature_matrix(X, expected_features, feature_names)
    selected_combo = combo or manifest.get("selected_combo")
    if not selected_combo:
        raise ValueError(
            "No final model combination is recorded. Specify one using the combo argument."
        )
    selected_keys = [part.strip() for part in selected_combo.split("+") if part.strip()]
    invalid = [key for key in selected_keys if key not in TRANSFER_MODEL_KEYS]
    if not selected_keys or invalid:
        raise ValueError(f"Unsupported model combination: {selected_combo}; invalid={invalid}")

    available = manifest.get("available_iterations", {})
    limit = n_iterations
    if limit is None:
        limit = manifest.get("selected_iterations")
    scaler_iterations = set(int(i) for i in manifest.get("scaler_iterations", []))
    model_iterations = {}
    model_tasks = []
    for model_key in selected_keys:
        iterations = [
            int(i) for i in available.get(model_key, [])
            if int(i) in scaler_iterations
        ]
        if limit is not None:
            iterations = iterations[: int(limit)]
        if not iterations:
            raise RuntimeError(
                f"No iterations available for the selected model ({model_key}): {stage_dir}"
            )
        model_iterations[model_key] = iterations
        model_tasks.extend((model_key, iteration) for iteration in iterations)

    torch_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else
        "cpu" if device == "auto" else device
    )
    probability_sum = np.zeros((X_aligned.shape[0], len(classes)), dtype=np.float64)
    probability_count = 0
    iterator = tqdm(
        model_tasks,
        desc=f"Frozen CODEX {manifest['stage']} inference",
        disable=not show_progress,
    )
    for model_key, iteration in iterator:
        out_dir = iteration_dir(stage_dir, iteration)
        scaler = joblib.load(out_dir / "scaler.joblib")
        X_scaled = np.ascontiguousarray(scaler.transform(X_aligned), dtype=np.float32)

        if model_key == "RF":
            rf_model = joblib.load(out_dir / "rf_model.joblib")
            try:
                import cupy as cp
                raw_probs = rf_model.predict_proba(cp.asarray(X_scaled))
            except Exception:
                raw_probs = rf_model.predict_proba(X_scaled)
            if hasattr(raw_probs, "get"):
                raw_probs = raw_probs.get()
            model_probs = _align_probability_columns(
                raw_probs,
                getattr(rf_model, "classes_", None),
                len(classes),
            )
            del rf_model, raw_probs
        elif model_key == "XGB":
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(str(out_dir / "xgb_model.json"))
            model_probs = _align_probability_columns(
                xgb_model.predict_proba(X_scaled),
                getattr(xgb_model, "classes_", None),
                len(classes),
            )
            del xgb_model
        elif model_key == "ENet":
            enet_model = joblib.load(out_dir / "enet_model.joblib")
            model_probs = _align_probability_columns(
                enet_model.predict_proba(X_scaled),
                getattr(enet_model, "classes_", None),
                len(classes),
            )
            del enet_model
        else:
            raw_probs, used_mlp_device = _predict_mlp(
                out_dir / "mlp_model.pt",
                X_scaled,
                device=torch_device,
                batch_size=batch_size,
            )
            # After the first CUDA failure, run remaining saved MLP iterations on CPU as well.
            torch_device = used_mlp_device
            model_probs = _align_probability_columns(raw_probs, None, len(classes))
            del raw_probs
        probability_sum += model_probs
        probability_count += 1
        del model_probs
        del scaler, X_scaled

    if probability_count == 0:
        raise RuntimeError(f"No model produced prediction probabilities: {stage_dir}")
    probabilities = (probability_sum / probability_count).astype(np.float32)
    labels = classes[np.argmax(probabilities, axis=1)]
    return {
        "labels": labels,
        "probabilities": pd.DataFrame(probabilities, columns=classes),
        "classes": classes,
        "iterations": sorted({i for values in model_iterations.values() for i in values}),
        "model_iterations": model_iterations,
        "model_keys": selected_keys,
        "selected_combo": "+".join(selected_keys),
        "mlp_inference_device": torch_device.type,
        "manifest": manifest,
    }


def predict_frozen_broad(
    X: pd.DataFrame | np.ndarray,
    transfer_root: os.PathLike[str] | str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply the V9 broad-lineage frozen ensemble."""
    return predict_frozen_stage(X, Path(transfer_root) / "broad", **kwargs)


def predict_frozen_subtypes(
    X: pd.DataFrame | np.ndarray,
    lineage_labels: Sequence[str],
    transfer_root: os.PathLike[str] | str,
    *,
    feature_names: Sequence[str] | None = None,
    unknown_label: str = "Unknown",
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply each lineage-specific subtype ensemble according to broad-lineage routing."""
    labels_array = np.asarray(lineage_labels, dtype=object)
    if len(labels_array) != len(X):
        raise ValueError("The length of lineage_labels must equal the number of rows in X.")

    subtype_root = Path(transfer_root) / "subtype"
    output_labels = np.full(len(labels_array), unknown_label, dtype=object)
    outputs: dict[str, Any] = {}
    for manifest_path in sorted(subtype_root.glob("*/manifest.json")):
        manifest = _read_json(manifest_path)
        lineage = str(manifest.get("lineage"))
        mask = labels_array == lineage
        if not np.any(mask):
            continue
        X_subset = X.iloc[mask] if isinstance(X, pd.DataFrame) else np.asarray(X)[mask]
        result = predict_frozen_stage(
            X_subset,
            manifest_path.parent,
            feature_names=feature_names,
            **kwargs,
        )
        output_labels[mask] = result["labels"]
        outputs[lineage] = result
    return {"labels": output_labels, "lineage_results": outputs}
