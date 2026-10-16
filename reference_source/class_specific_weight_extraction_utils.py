#!/usr/bin/env python3
"""Final-owned frozen-model helpers for class-specific marker-weight extraction.

This module reads the exact saved broad/subtype XGB+MLP iteration artifacts and
supports patient-held-out permutation attribution. It contains no scRNA
centroid fitting or classifier-transfer validation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


FINAL_DIR = Path(__file__).resolve().parent.parent
REVIEW_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = FINAL_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from V9_transfer_utils import (  # noqa: E402
    _align_probability_columns,
    _predict_mlp,
    iteration_dir,
)


CODEX_STEP02_H5AD = FINAL_DIR / "checkpoints" / "mv1" / "mv1_step02.h5ad"
CODEX_STEP04B_H5AD = FINAL_DIR / "checkpoints" / "mv1" / "mv1_step04B_pan_cell.h5ad"
TRANSFER_ROOT = FINAL_DIR / "checkpoints" / "transfer_models"
SC_RNA_ROOT = FINAL_DIR / "checkpoints" / "scRNA"
ROIDER_FEATURE_H5AD = SC_RNA_ROOT / "Roider" / "01_roider_codex_proxy_features.h5ad"
GSE_FEATURE_H5AD = SC_RNA_ROOT / "GSE182434" / "01_gse182434_codex_proxy_features.h5ad"
ROIDER_COVERAGE_CSV = FINAL_DIR / "tables" / "scRNA" / "scRNA_Roider" / "tables" / "codex_gene_proxy_coverage.csv"
GSE_COVERAGE_CSV = FINAL_DIR / "tables" / "scRNA" / "scRNA_GSE182434" / "tables" / "codex_gene_proxy_coverage.csv"
MLP_CUDA_WORKER = REVIEW_ROOT / "class_specific_mlp_cuda_worker.py"

RANDOM_SEED = 42
CV_TEST_SIZE = 0.30
MODEL_KEYS = ("XGB", "MLP")


@dataclass(frozen=True)
class Config:
    importance_iterations: int | None
    permutation_repeats: int
    max_cells_per_class_patient: int
    random_seed: int
    device: str
    batch_size: int
    hash_inputs: bool
    synthetic_smoke: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--importance-iterations",
        type=int,
        default=100,
        help="Locked to the exact selected iterations 1 through 100.",
    )
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--max-cells-per-class-patient", type=int, default=250)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--skip-input-hash", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument(
        "--cuda-preflight",
        action="store_true",
        help="Run an isolated frozen-MLP CUDA/CUBLAS/model equivalence check without creating a review run.",
    )
    args = parser.parse_args()
    for name in ("importance_iterations", "permutation_repeats", "max_cells_per_class_patient", "batch_size"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.random_seed != RANDOM_SEED:
        parser.error(f"--random-seed is locked to {RANDOM_SEED} for the final reproducible analysis")
    locked = {
        "importance_iterations": 100,
        "permutation_repeats": 10,
        "max_cells_per_class_patient": 250,
        "batch_size": 8192,
    }
    for name, expected in locked.items():
        if getattr(args, name) != expected:
            parser.error(f"--{name.replace('_', '-')} is locked to {expected}")
    if args.device != "cuda":
        parser.error("--device is locked to cuda; CPU/auto fallback is prohibited")
    if args.skip_input_hash:
        parser.error("--skip-input-hash is prohibited for the final class-specific run")
    return args


def lock_randomness(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [json_ready(v) for v in list(value)]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class CudaMLPWorker:
    """Persistent CUDA-only worker isolated from XGBoost and AnnData imports."""

    def __init__(
        self,
        seed: int,
        exchange_parent: Path,
        response_timeout: float = 600.0,
    ) -> None:
        if not MLP_CUDA_WORKER.is_file():
            raise FileNotFoundError(MLP_CUDA_WORKER)
        self.response_timeout = float(response_timeout)
        self.exchange_parent = exchange_parent.resolve()
        allowed_roots = (REVIEW_ROOT.resolve(), Path("/dev/shm").resolve())
        if not any(
            self.exchange_parent == root or root in self.exchange_parent.parents
            for root in allowed_roots
        ):
            raise ValueError(
                f"CUDA exchange parent must be under one of {allowed_roots}: "
                f"{self.exchange_parent}"
            )
        self.exchange_parent.mkdir(parents=True, exist_ok=True)
        self.exchange_root = Path(
            tempfile.mkdtemp(prefix="roider_mlp_cuda_", dir=self.exchange_parent)
        ).resolve()
        self.stderr_path = self.exchange_root / "worker_stderr.log"
        self.stderr_handle = self.stderr_path.open("w+", encoding="utf-8")
        self.process: subprocess.Popen[str] | None = None
        self.counter = 0
        self.info: dict[str, Any] = {}
        try:
            environment = os.environ.copy()
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(MLP_CUDA_WORKER),
                    "--exchange-root",
                    str(self.exchange_root),
                    "--seed",
                    str(int(seed)),
                ],
                cwd=REVIEW_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr_handle,
                text=True,
                bufsize=1,
                env=environment,
            )
            response = self._read_response(timeout=120.0)
            if response.get("status") != "ready" or response.get("device") != "cuda":
                raise RuntimeError(f"CUDA MLP worker preflight failed: {response}")
            self.info = response
        except Exception:
            self.close()
            raise

    def _stderr_text(self) -> str:
        try:
            self.stderr_handle.flush()
            return self.stderr_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        except Exception:
            return ""

    def _read_response(self, timeout: float | None = None) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("CUDA MLP worker is not running")
        deadline = time.monotonic() + (self.response_timeout if timeout is None else float(timeout))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for CUDA MLP worker. stderr={self._stderr_text()}"
                )
            readable, _, _ = select.select([self.process.stdout], [], [], min(1.0, remaining))
            if readable:
                line = self.process.stdout.readline()
                if line:
                    return json.loads(line)
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"CUDA MLP worker exited with code {return_code}. stderr={self._stderr_text()}"
                )

    def predict(self, model_path: Path, x_scaled: np.ndarray, batch_size: int) -> np.ndarray:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("CUDA MLP worker is not running")
        if self.process.poll() is not None:
            raise RuntimeError(
                f"CUDA MLP worker is not alive. stderr={self._stderr_text()}"
            )
        self.counter += 1
        input_path = self.exchange_root / f"request_{self.counter:08d}_input.npy"
        output_path = self.exchange_root / f"request_{self.counter:08d}_output.npy"
        try:
            with input_path.open("wb") as handle:
                np.save(handle, np.ascontiguousarray(x_scaled, dtype=np.float32))
            request = {
                "op": "predict",
                "model_path": str(model_path.resolve()),
                "input_path": str(input_path),
                "output_path": str(output_path),
                "batch_size": int(batch_size),
            }
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            response = self._read_response()
            if response.get("status") != "ok" or response.get("device") != "cuda":
                raise RuntimeError(
                    f"CUDA MLP inference failed without CPU fallback: {response}; "
                    f"stderr={self._stderr_text()}"
                )
            probabilities = np.load(output_path, allow_pickle=False)
            return np.ascontiguousarray(probabilities, dtype=np.float32)
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def clear_model_cache(self) -> int:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("CUDA MLP worker is not running")
        if self.process.poll() is not None:
            raise RuntimeError(
                f"CUDA MLP worker is not alive. stderr={self._stderr_text()}"
            )
        self.process.stdin.write(json.dumps({"op": "clear_model_cache"}) + "\n")
        self.process.stdin.flush()
        response = self._read_response()
        if response.get("status") != "ok" or response.get("device") != "cuda":
            raise RuntimeError(
                f"CUDA MLP cache clear failed: {response}; stderr={self._stderr_text()}"
            )
        if int(response.get("model_cache_size", -1)) != 0:
            raise RuntimeError(f"CUDA MLP cache was not cleared: {response}")
        return int(response.get("cleared_models", 0))

    def close(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    process.stdin.flush()
                self._read_response(timeout=10.0)
            except Exception:
                process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
        try:
            self.stderr_handle.close()
        except Exception:
            pass
        if self.exchange_root.exists():
            shutil.rmtree(self.exchange_root)
        try:
            self.exchange_parent.rmdir()
        except OSError:
            pass


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, with_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path) if with_hash else None,
    }


def assert_under(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def prepare_run_dirs(run_id: str) -> dict[str, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(
            "run_id must contain only letters, digits, dot, underscore, or hyphen "
            "and may not contain path separators"
        )
    runs_root = (REVIEW_ROOT / "runs").resolve()
    run_root = assert_under(runs_root / run_id, runs_root)
    if run_root.exists():
        raise FileExistsError(f"Run directory already exists: {run_root}")
    names = ["code", "notebooks", "manifests", "weights", "caches", "tables", "figures", "logs"]
    paths = {"run": run_root}
    for name in names:
        path = assert_under(run_root / name, run_root)
        path.mkdir(parents=True, exist_ok=False)
        paths[name] = path
    return paths


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("scrna_option2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def read_manifest(stage_dir: Path) -> dict[str, Any]:
    path = stage_dir / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Frozen stage is not complete: {path}")
    if payload.get("selected_combo") != "XGB+MLP":
        raise RuntimeError(f"Expected XGB+MLP, found {payload.get('selected_combo')}: {path}")
    return payload


def evenly_spaced_iterations(manifest: dict[str, Any], requested_n: int | None) -> list[int]:
    available = set(map(int, manifest["available_iterations"]["XGB"]))
    available &= set(map(int, manifest["available_iterations"]["MLP"]))
    available &= set(map(int, manifest.get("scaler_iterations", [])))
    selected_limit = int(manifest["selected_iterations"])
    ordered = sorted(i for i in available if i <= selected_limit)
    if requested_n is None or requested_n >= len(ordered):
        return ordered
    positions = np.linspace(0, len(ordered) - 1, num=requested_n)
    chosen = sorted({ordered[int(round(pos))] for pos in positions})
    if len(chosen) != requested_n:
        for value in ordered:
            if value not in chosen:
                chosen.append(value)
            if len(chosen) == requested_n:
                break
        chosen = sorted(chosen)
    return chosen


def valid_label_mask(values: Sequence[Any]) -> tuple[np.ndarray, pd.Series]:
    labels = pd.Series(values).astype("string").str.strip()
    invalid = labels.isna() | labels.eq("") | labels.str.fullmatch(r"(unknown|na|nan)", case=False, na=False)
    return (~invalid).to_numpy(dtype=bool), labels


def load_reference_stage(stage: str) -> dict[str, Any]:
    if stage == "broad":
        path = CODEX_STEP02_H5AD
        adata = ad.read_h5ad(path, backed="r")
        disease = adata.obs["Disease"].astype("string").str.strip()
        ref_mask = ~disease.eq("DLBCL").fillna(False).to_numpy(dtype=bool)
        raw = adata.obs.loc[ref_mask, "GT_Lineage_Broad"].to_numpy()
        valid, clean = valid_label_mask(raw)
        ref_indices = np.flatnonzero(ref_mask)[valid]
        labels = clean[valid].astype(str).to_numpy()
        groups = adata.obs.iloc[ref_indices]["PatientID"].astype("string").to_numpy(dtype=str)
        label_encoder = LabelEncoder().fit(labels)
        y = label_encoder.transform(labels)
        stage_dir = TRANSFER_ROOT / "broad"
    elif stage == "tcell_subtype":
        path = CODEX_STEP04B_H5AD
        adata = ad.read_h5ad(path, backed="r")
        disease = adata.obs["Disease"].astype("string")
        ref_mask = (~disease.str.contains("DLBCL", na=False)) & adata.obs["Modeling_Broad"].astype("string").eq("T cells")
        ref_indices = np.flatnonzero(ref_mask.to_numpy(dtype=bool))
        labels = adata.obs.iloc[ref_indices]["GT_Subtype"].astype(str).to_numpy()
        groups = adata.obs.iloc[ref_indices]["PatientID"].astype("string").to_numpy(dtype=str)
        label_encoder = LabelEncoder().fit(labels)
        y = label_encoder.transform(labels)
        stage_dir = TRANSFER_ROOT / "subtype" / "T_cells"
    else:
        raise ValueError(stage)

    manifest = read_manifest(stage_dir)
    if list(label_encoder.classes_) != list(manifest["classes"]):
        adata.file.close()
        raise RuntimeError(
            f"Class order mismatch for {stage}: reconstructed={list(label_encoder.classes_)} "
            f"manifest={manifest['classes']}"
        )
    feature_names = list(map(str, manifest["feature_names"]))
    var_names = list(map(str, adata.var_names))
    missing = [feature for feature in feature_names if feature not in var_names]
    if missing:
        adata.file.close()
        raise KeyError(f"Missing CODEX features in {path}: {missing}")
    feature_idx = np.asarray([var_names.index(feature) for feature in feature_names], dtype=int)
    return {
        "stage": stage,
        "path": path,
        "adata": adata,
        "ref_indices": ref_indices,
        "labels": labels,
        "y": y,
        "groups": groups,
        "classes": list(label_encoder.classes_),
        "feature_names": feature_names,
        "feature_idx": feature_idx,
        "stage_dir": stage_dir,
        "manifest": manifest,
    }


def read_backed_rows(adata: ad.AnnData, rows: np.ndarray, feature_idx: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=int)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    matrix = adata.X[sorted_rows]
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)[:, feature_idx]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.ascontiguousarray(matrix[inverse], dtype=np.float32)


def reconstruct_split(reference: dict[str, Any], iteration: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=CV_TEST_SIZE, random_state=RANDOM_SEED + iteration - 1)
    train_idx, validation_idx = next(
        splitter.split(np.zeros(len(reference["y"])), reference["y"], groups=reference["groups"])
    )
    meta_path = iteration_dir(reference["stage_dir"], iteration) / "iteration.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    observed = {
        "train_n": int(len(train_idx)),
        "validation_n": int(len(validation_idx)),
        "train_patient_n": int(pd.Series(reference["groups"][train_idx]).nunique()),
        "validation_patient_n": int(pd.Series(reference["groups"][validation_idx]).nunique()),
    }
    for key, value in observed.items():
        expected = meta.get(key)
        if expected is not None and int(expected) != value:
            raise RuntimeError(
                f"Split reconstruction mismatch: stage={reference['stage']} iter={iteration} "
                f"{key} observed={value} expected={expected}"
            )
    if set(reference["groups"][train_idx]) & set(reference["groups"][validation_idx]):
        raise RuntimeError(f"Train/validation group overlap: {reference['stage']} iter={iteration}")
    return train_idx, validation_idx, {**meta, **observed}


def stratified_cap(indices: np.ndarray, y: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    selected: list[int] = []
    for label in np.unique(y[indices]):
        label_idx = indices[y[indices] == label]
        if len(label_idx) > cap:
            label_idx = np.sort(rng.choice(label_idx, size=cap, replace=False))
        selected.extend(map(int, label_idx))
    return np.asarray(sorted(selected), dtype=int)


def balanced_multiclass_loss(y: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    sample_weight = np.asarray([1.0 / counts[label] for label in y], dtype=float)
    return float(log_loss(y, probabilities, labels=np.arange(n_classes), sample_weight=sample_weight))


def predict_iteration_model(
    stage_dir: Path,
    iteration: int,
    model_key: str,
    x_raw: np.ndarray,
    n_classes: int,
    device: torch.device,
    batch_size: int,
    xgb_cache: dict[tuple[str, int], xgb.XGBClassifier] | None = None,
    mlp_worker: CudaMLPWorker | None = None,
) -> tuple[np.ndarray, torch.device]:
    out_dir = iteration_dir(stage_dir, iteration)
    scaler = joblib.load(out_dir / "scaler.joblib")
    x_scaled = np.ascontiguousarray(scaler.transform(x_raw), dtype=np.float32)
    if model_key == "XGB":
        cache_key = (str(stage_dir), int(iteration))
        model = xgb_cache.get(cache_key) if xgb_cache is not None else None
        if model is None:
            model = xgb.XGBClassifier()
            model.load_model(str(out_dir / "xgb_model.json"))
            if xgb_cache is not None:
                xgb_cache[cache_key] = model
        probabilities = _align_probability_columns(
            model.predict_proba(x_scaled), getattr(model, "classes_", None), n_classes
        )
        return probabilities, device
    if model_key == "MLP":
        if device.type == "cuda":
            if mlp_worker is None:
                raise RuntimeError("CUDA was requested but the isolated MLP worker is unavailable")
            raw = mlp_worker.predict(out_dir / "mlp_model.pt", x_scaled, batch_size)
            return _align_probability_columns(raw, None, n_classes), device
        raw, used_device = _predict_mlp(
            out_dir / "mlp_model.pt",
            x_scaled,
            device=device,
            batch_size=batch_size,
        )
        return _align_probability_columns(raw, None, n_classes), used_device
    raise ValueError(model_key)


def permutation_stack(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_cells, n_features = x.shape
    stacked = np.tile(x, (n_features, 1))
    for feature_idx in range(n_features):
        start = feature_idx * n_cells
        stop = start + n_cells
        stacked[start:stop, feature_idx] = rng.permutation(x[:, feature_idx])
    return np.ascontiguousarray(stacked, dtype=np.float32)


def load_proxy_groups(coverage_path: Path, feature_names: Sequence[str]) -> pd.DataFrame:
    coverage = pd.read_csv(coverage_path)
    coverage = coverage.set_index("codex_marker").loc[list(feature_names)].reset_index()
    coverage["included"] = coverage["covered"].map(
        lambda value: value if isinstance(value, (bool, np.bool_)) else str(value).strip().lower() == "true"
    )
    coverage["proxy_group"] = coverage["present_genes"].fillna("").astype(str).map(
        lambda value: "|".join(sorted(filter(None, value.split("|"))))
    )
    coverage["included"] &= coverage["proxy_group"].ne("")
    if int(coverage["included"].sum()) < 2:
        raise RuntimeError(f"Too few covered scRNA proxy markers: {coverage_path}")
    group_counts = coverage.loc[coverage["included"]].groupby("proxy_group")["codex_marker"].transform("count")
    coverage["duplicate_proxy_group"] = False
    coverage.loc[coverage["included"], "duplicate_proxy_group"] = group_counts > 1
    return coverage


def collapse_proxy_matrix(
    obs: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    groups = []
    columns = []
    audit_rows = []
    for row in coverage.loc[~coverage["included"]].itertuples(index=False):
        audit_rows.append({
            "proxy_group": "__uncovered__",
            "markers": str(row.codex_marker),
            "n_markers": 1,
            "included": False,
            "collapse_rule": "excluded because no scRNA gene proxy was present; remaining weights renormalized",
            "max_within_group_abs_difference": math.nan,
        })
    included = coverage[coverage["included"]]
    for proxy_group, group_df in included.groupby("proxy_group", sort=True):
        markers = group_df["codex_marker"].astype(str).tolist()
        marker_cols = [f"codex_{marker}" for marker in markers]
        values = obs[marker_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        columns.append(values.mean(axis=1))
        groups.append(proxy_group)
        audit_rows.append({
            "proxy_group": proxy_group,
            "markers": "|".join(markers),
            "n_markers": len(markers),
            "included": True,
            "collapse_rule": "mean feature value; sum protein-space weights then renormalize",
            "max_within_group_abs_difference": float(np.max(np.ptp(values, axis=1))) if values.shape[1] > 1 else 0.0,
        })
    return np.column_stack(columns).astype(np.float32), groups, pd.DataFrame(audit_rows)


def subtype_truth(dataset_name: str, obs: pd.DataFrame) -> pd.Series:
    raw = obs["reference_population_raw"].astype("string")
    if dataset_name == "Roider":
        mapping = {"TH": "TH", "TFH": "TFH", "TREG": "TREG"}
    else:
        mapping = {
            "T cells CD8": "CD8T",
            "T cells CD4": "TH",
            "TFH": "TFH",
            "Tregs": "TREG",
        }
    return raw.map(mapping).astype("string")


def evaluate_predictions(
    truth: pd.Series,
    predicted: pd.Series,
    classes: Sequence[str],
    dataset: str,
    task: str,
    method: str,
    group: str,
) -> dict[str, Any]:
    mask = truth.notna() & truth.isin(classes)
    y_true = truth[mask].astype(str).to_numpy()
    y_pred = predicted[mask].astype(str).to_numpy()
    if len(y_true) == 0:
        return {
            "dataset": dataset, "task": task, "method": method, "group": group,
            "n_cells": 0, "balanced_accuracy": np.nan,
            "macro_f1_present": np.nan, "macro_f1_model_space": np.nan,
        }
    present = sorted(pd.unique(y_true))
    return {
        "dataset": dataset,
        "task": task,
        "method": method,
        "group": group,
        "n_cells": int(len(y_true)),
        "classes": "|".join(map(str, classes)),
        "present_truth_classes": "|".join(present),
        "predictions_outside_class_space": int((~pd.Series(y_pred).isin(classes)).sum()),
        "balanced_accuracy": float(
            np.mean([np.mean(y_pred[y_true == class_name] == class_name) for class_name in present])
        ),
        "macro_f1_present": float(
            f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
        ),
        "macro_f1_model_space": float(
            f1_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0)
        ),
    }


def bootstrap_mean_ci(values: Sequence[float], seed: int, n_boot: int = 10_000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(n_boot, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def summarize_group_metrics(metrics: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    group_metrics = metrics[~metrics["group"].eq("__pooled__")].copy()
    rows: list[dict[str, Any]] = []
    metric_columns = ["balanced_accuracy", "macro_f1_present", "macro_f1_model_space"]
    for (dataset, task, method), frame in group_metrics.groupby(["dataset", "task", "method"], sort=True):
        for metric_idx, metric in enumerate(metric_columns):
            values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(
                values,
                random_seed + metric_idx * 10_003 + sum(map(ord, f"{dataset}:{task}:{method}")),
            )
            rows.append({
                "dataset": dataset,
                "task": task,
                "method": method,
                "metric": metric,
                "n_groups": int(len(values)),
                "group_mean": float(np.mean(values)) if len(values) else math.nan,
                "group_median": float(np.median(values)) if len(values) else math.nan,
                "group_bootstrap_ci95_low": low,
                "group_bootstrap_ci95_high": high,
            })
    return pd.DataFrame(rows)


def build_pooled_class_diagnostics(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = metrics[metrics["group"].eq("__pooled__")]
    confusion_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for metric_row in pooled.itertuples(index=False):
        classes = str(metric_row.classes).split("|")
        frame = predictions[
            predictions["dataset"].eq(metric_row.dataset)
            & predictions["task"].eq(metric_row.task)
            & predictions["method"].eq(metric_row.method)
        ].copy()
        frame = frame[frame["reference_label"].isin(classes)]
        truth = frame["reference_label"].astype(str)
        predicted = frame["prediction"].astype(str)
        prediction_classes = classes + sorted(set(predicted) - set(classes))
        table = pd.crosstab(truth, predicted).reindex(index=classes, columns=prediction_classes, fill_value=0)
        for true_class in classes:
            for predicted_class in prediction_classes:
                confusion_rows.append({
                    "dataset": metric_row.dataset,
                    "task": metric_row.task,
                    "method": metric_row.method,
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "n_cells": int(table.loc[true_class, predicted_class]),
                })
        for class_name in classes:
            tp = int(((truth == class_name) & (predicted == class_name)).sum())
            fn = int(((truth == class_name) & (predicted != class_name)).sum())
            fp = int(((truth != class_name) & (predicted == class_name)).sum())
            recall = tp / (tp + fn) if tp + fn else math.nan
            precision = tp / (tp + fp) if tp + fp else math.nan
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
            class_rows.append({
                "dataset": metric_row.dataset,
                "task": metric_row.task,
                "method": metric_row.method,
                "class": class_name,
                "support": tp + fn,
                "true_positive": tp,
                "false_negative": fn,
                "false_positive": fp,
                "recall": recall,
                "precision": precision,
                "f1": f1,
            })
    return pd.DataFrame(confusion_rows), pd.DataFrame(class_rows)


def output_inventory(run_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            records.append(file_record(path, with_hash=True))
    return records


def selected_input_paths(
    references: Sequence[dict[str, Any]],
    stage_iterations: dict[str, Sequence[int]],
) -> list[Path]:
    paths = [
        CODEX_STEP02_H5AD,
        CODEX_STEP04B_H5AD,
        ROIDER_FEATURE_H5AD,
        GSE_FEATURE_H5AD,
        ROIDER_COVERAGE_CSV,
        GSE_COVERAGE_CSV,
    ]
    for reference in references:
        paths.append(reference["stage_dir"] / "manifest.json")
        for iteration in stage_iterations[reference["stage"]]:
            out_dir = iteration_dir(reference["stage_dir"], iteration)
            paths.extend([
                out_dir / "iteration.json",
                out_dir / "scaler.joblib",
                out_dir / "xgb_model.json",
                out_dir / "mlp_model.pt",
            ])
    return list(dict.fromkeys(paths))


def build_model_artifact_audit(
    references: Sequence[dict[str, Any]],
    stage_iterations: dict[str, Sequence[int]],
    input_records: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    record_lookup = {record["path"]: record for record in input_records}
    rows: list[dict[str, Any]] = []
    for reference in references:
        manifest = reference["manifest"]
        for iteration in stage_iterations[reference["stage"]]:
            out_dir = iteration_dir(reference["stage_dir"], iteration)
            scaler_path = (out_dir / "scaler.joblib").resolve()
            iteration_path = (out_dir / "iteration.json").resolve()
            for model_key, filename in [("XGB", "xgb_model.json"), ("MLP", "mlp_model.pt")]:
                model_path = (out_dir / filename).resolve()
                rows.append({
                    "stage": reference["stage"],
                    "iteration": int(iteration),
                    "model": model_key,
                    "selected_combo": manifest["selected_combo"],
                    "selected_iterations": int(manifest["selected_iterations"]),
                    "probability_rule": manifest["probability_rule"],
                    "feature_order_source": str((reference["stage_dir"] / "manifest.json").resolve()),
                    "model_path": str(model_path),
                    "model_sha256": record_lookup[str(model_path)]["sha256"],
                    "scaler_path": str(scaler_path),
                    "scaler_sha256": record_lookup[str(scaler_path)]["sha256"],
                    "iteration_metadata_path": str(iteration_path),
                    "iteration_metadata_sha256": record_lookup[str(iteration_path)]["sha256"],
                    "used_for_attribution": True,
                })
    return pd.DataFrame(rows)


def run_cuda_preflight(random_seed: int, batch_size: int) -> dict[str, Any]:
    """Verify isolated CUDA matmul and exact frozen-model inference without a run write."""
    stage_dir = TRANSFER_ROOT / "broad"
    manifest = read_manifest(stage_dir)
    out_dir = iteration_dir(stage_dir, 1)
    scaler = joblib.load(out_dir / "scaler.joblib")
    feature_count = len(manifest["feature_names"])
    rng = np.random.default_rng(random_seed)
    center = np.asarray(getattr(scaler, "mean_", np.zeros(feature_count)), dtype=np.float32)
    scale = np.asarray(getattr(scaler, "scale_", np.ones(feature_count)), dtype=np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0).astype(np.float32)
    x_raw = center + rng.normal(size=(8192, feature_count)).astype(np.float32) * scale
    x_scaled = np.ascontiguousarray(scaler.transform(x_raw), dtype=np.float32)

    worker = CudaMLPWorker(random_seed, REVIEW_ROOT / ".worker_tmp")
    try:
        model_path = out_dir / "mlp_model.pt"
        first = worker.predict(model_path, x_scaled, batch_size)
        second = worker.predict(model_path, x_scaled, batch_size)
        cpu, _ = _predict_mlp(
            model_path,
            x_scaled,
            device=torch.device("cpu"),
            batch_size=batch_size,
        )
        expected_shape = (len(x_scaled), len(manifest["classes"]))
        if first.shape != expected_shape:
            raise RuntimeError(f"CUDA preflight shape mismatch: {first.shape} != {expected_shape}")
        if not np.isfinite(first).all():
            raise RuntimeError("CUDA preflight produced non-finite probabilities")
        if not np.allclose(first.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
            raise RuntimeError("CUDA preflight probability rows do not sum to one")
        if not np.array_equal(first, second):
            raise RuntimeError("Repeated deterministic CUDA inference was not bitwise identical")
        max_abs_cpu_delta = float(np.max(np.abs(first - cpu)))
        if not np.allclose(first, cpu, rtol=1e-4, atol=1e-5):
            raise RuntimeError(
                f"CUDA probabilities differ from the CPU frozen-model reference: max_abs={max_abs_cpu_delta}"
            )
        argmax_agreement = float(np.mean(np.argmax(first, axis=1) == np.argmax(cpu, axis=1)))
        if argmax_agreement != 1.0:
            raise RuntimeError(f"CUDA/CPU argmax mismatch: agreement={argmax_agreement}")
        return {
            **worker.info,
            "status": "cuda_preflight_complete",
            "model_path": str(model_path.resolve()),
            "scaler_path": str((out_dir / "scaler.joblib").resolve()),
            "input_shape": list(x_scaled.shape),
            "output_shape": list(first.shape),
            "probability_rows_sum_to_one": True,
            "repeat_bitwise_identical": True,
            "cpu_reference_max_abs_delta": max_abs_cpu_delta,
            "cpu_reference_argmax_agreement": argmax_agreement,
        }
    finally:
        worker.close()
