#!/usr/bin/env python3
"""Persistent CUDA-only inference worker for frozen Roider MLP packages.

This process intentionally imports no XGBoost, AnnData, or scRNA libraries.
It keeps PyTorch CUDA isolated from the review backend, caches frozen MLP
packages for repeated permutation inference, and never falls back to CPU.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn as nn


MODEL_CACHE_SIZE = 100


class SimpleMLP(nn.Module):
    """Inference architecture matching the saved V9 SimpleMLP package."""

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


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def configure_cuda(seed: int) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the isolated MLP worker")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")

    left = torch.linspace(-1.0, 1.0, 64 * 64, device=device).reshape(64, 64)
    right = torch.linspace(1.0, -1.0, 64 * 64, device=device).reshape(64, 64)
    probe = left @ right
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(probe).all().item()):
        raise RuntimeError("CUDA cuBLAS preflight returned non-finite values")
    del left, right, probe
    torch.cuda.empty_cache()
    return device


def load_torch_package(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def path_under_root(value: str, root: Path) -> Path:
    path = Path(value).resolve()
    path.relative_to(root)
    return path


class FrozenMLPRunner:
    def __init__(self, device: torch.device, exchange_root: Path) -> None:
        self.device = device
        self.exchange_root = exchange_root.resolve()
        self.model_path: Path | None = None
        self.model: SimpleMLP | None = None
        self.input_dim: int | None = None
        self.n_classes: int | None = None
        self.model_cache: OrderedDict[
            Path, tuple[SimpleMLP, int, int]
        ] = OrderedDict()

    def load_model(self, model_path: Path) -> bool:
        resolved = model_path.resolve()
        cached = self.model_cache.get(resolved)
        if cached is not None:
            self.model_cache.move_to_end(resolved)
            self.model, self.input_dim, self.n_classes = cached
            self.model_path = resolved
            return True
        package = load_torch_package(resolved)
        input_dim = int(package["input_dim"])
        n_classes = int(package["n_classes"])
        model = SimpleMLP(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dims=package.get("hidden_dims", [256, 128]),
            dropout_rate=float(package.get("dropout_rate", 0.5)),
        ).to(self.device)
        model.load_state_dict(package["state_dict"])
        model.eval()
        self.model = model
        self.model_path = resolved
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.model_cache[resolved] = (model, input_dim, n_classes)
        if len(self.model_cache) > MODEL_CACHE_SIZE:
            self.model_cache.popitem(last=False)
        torch.cuda.synchronize(self.device)
        return False

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        input_path = path_under_root(str(request["input_path"]), self.exchange_root)
        output_path = path_under_root(str(request["output_path"]), self.exchange_root)
        model_path = Path(str(request["model_path"]))
        batch_size = int(request["batch_size"])
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        cache_hit = self.load_model(model_path)
        assert self.model is not None and self.input_dim is not None and self.n_classes is not None

        array = np.load(input_path, mmap_mode="r")
        if array.ndim != 2 or int(array.shape[1]) != self.input_dim:
            raise ValueError(
                f"Frozen MLP input mismatch: observed={array.shape}, expected=(*, {self.input_dim})"
            )
        probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, int(array.shape[0]), batch_size):
                batch_np = np.ascontiguousarray(array[start : start + batch_size], dtype=np.float32)
                batch = torch.as_tensor(batch_np, dtype=torch.float32, device=self.device)
                probabilities.append(torch.softmax(self.model(batch), dim=1).cpu().numpy())
        torch.cuda.synchronize(self.device)
        result = np.concatenate(probabilities, axis=0).astype(np.float32, copy=False)
        if result.shape != (int(array.shape[0]), self.n_classes):
            raise RuntimeError(f"Frozen MLP output mismatch: observed={result.shape}")
        if not np.isfinite(result).all():
            raise RuntimeError("Frozen MLP CUDA inference returned non-finite probabilities")

        temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_output.open("wb") as handle:
            np.save(handle, result)
        os.replace(temporary_output, output_path)
        return {
            "status": "ok",
            "shape": list(result.shape),
            "device": "cuda",
            "model_path": str(self.model_path),
            "model_cache_hit": bool(cache_hit),
            "model_cache_size": len(self.model_cache),
        }

    def clear_model_cache(self) -> dict[str, Any]:
        cached_models = len(self.model_cache)
        self.model = None
        self.model_path = None
        self.input_dim = None
        self.n_classes = None
        self.model_cache.clear()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        return {
            "status": "ok",
            "device": "cuda",
            "cleared_models": cached_models,
            "model_cache_size": 0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exchange_root = Path(args.exchange_root).resolve()
    if not exchange_root.is_dir():
        raise FileNotFoundError(exchange_root)
    device = configure_cuda(args.seed)
    runner = FrozenMLPRunner(device, exchange_root)
    emit({
        "status": "ready",
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": int(args.seed),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "matmul_preflight": "passed",
        "model_cache_capacity": MODEL_CACHE_SIZE,
    })

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            operation = request.get("op")
            if operation == "shutdown":
                emit({"status": "bye"})
                return 0
            if operation == "clear_model_cache":
                emit(runner.clear_model_cache())
                continue
            if operation != "predict":
                raise ValueError(f"Unsupported worker operation: {operation}")
            emit(runner.predict(request))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit({"status": "error", "error": repr(exc)})
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
