#!/usr/bin/env python3
"""Private implementation for pinned frozen-signature input validation.

This module never trains a model, never chooses a latest run for the canonical
workflow, and never edits an attribution run.  The saved
`class_normalized_importance` from the pinned Step02 broad run and pinned
Step04B subtype run supplies the scRNA signature weight. Paired attribution
frequency is joined only as stability metadata and is never multiplied into or
substituted for the weight itself. The directional consumer derives the sign
only from the completed DLBCL q10-to-q90 ALE response bundle. It never interprets
the generic saved ``rank_within_class`` field as a magnitude rank.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parent
FINAL_ROOT = WORKFLOW_ROOT.parent
SOURCE_INPUTS_ROOT = FINAL_ROOT / "tables" / "scRNA" / "fixed_signature" / "source_inputs"
STEP04B_ROOT = SOURCE_INPUTS_ROOT / "step04B"
RUNS_ROOT = STEP04B_ROOT / "step04b_joint_importance_runs"
STEP02_ROOT = SOURCE_INPUTS_ROOT / "step02"
STEP02_RUNS_ROOT = STEP02_ROOT / "step02_broad_attribution_runs"
STEP02_ALE_ROOT = STEP02_ROOT / "dlbcl_marker_response_xgb_mlp"
STEP04B_ALE_ROOT = STEP04B_ROOT / "dlbcl_marker_response_xgb_mlp"
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
EXPECTED_FEATURE_ROWS = 650
EXPECTED_FEATURES = 50
EXPECTED_CLASSES = 13
EXPECTED_BROAD_FEATURE_ROWS = 250
EXPECTED_BROAD_CLASSES = 5
DIRECTION_CONSISTENCY_THRESHOLD = 0.8
DIRECTION_ZERO_TOLERANCE = 1e-12
BROAD_CLASS_ORDER = [
    "B cells",
    "Myeloid cells",
    "NK cells",
    "Stromal cells",
    "T cells",
]
TCELL_CLASS_DISPLAY = {
    "Th": "TH",
    "Tfh": "TFH",
    "Treg": "TREG",
    "CD8T": "CD8T",
    "CD4_CTL": "CD4_CTL",
    "Tpr": "TPR",
}

COVERAGE_PATHS = {
    "Roider": (
        FINAL_ROOT
        / "tables"
        / "scRNA"
        / "scRNA_Roider"
        / "tables"
        / "codex_gene_proxy_coverage.csv"
    ),
    "GSE182434": (
        FINAL_ROOT
        / "tables"
        / "scRNA"
        / "scRNA_GSE182434"
        / "tables"
        / "codex_gene_proxy_coverage.csv"
    ),
}


@dataclass(frozen=True)
class Step04BRun:
    run_id: str
    root: Path
    manifest: dict[str, Any]
    method: str
    completed_at: str


@dataclass(frozen=True)
class Step02Run:
    run_id: str
    root: Path
    manifest: dict[str, Any]
    completed_at: str


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").str.strip().str.lower().eq("true")



def _validated_ale_summary(
    response_root: Path,
    *,
    expected_rows: int,
    expected_classes: int,
    source_label: str,
) -> pd.DataFrame:
    """Validate one completed DLBCL q10-to-q90 ALE response bundle."""

    response_root = Path(response_root).resolve()
    manifest_path = response_root / "manifests" / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    effective_complete = bool(
        manifest.get("status") == "complete" and manifest.get("input_unchanged") is True
    )
    if not effective_complete:
        recovery_path = response_root / "manifests" / "finalization_recovery_manifest.json"
        if recovery_path.is_file():
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            effective_complete = bool(
                recovery.get("status") == "complete"
                and recovery.get("calculation_status") == "complete_100_of_100_pairs"
                and recovery.get("calculation_reexecuted") is False
                and recovery.get("input_unchanged") is True
                and recovery.get("validation", {}).get("all_pass") is True
            )
    direction = manifest.get("direction_contract", {})
    if (
        not effective_complete
        or manifest.get("retraining") is not False
        or manifest.get("source_models_read_only") is not True
        or direction.get("metric")
        != "patient-equal first-order ALE q10-to-q90 contrast"
        or direction.get("consistency")
        != "minimum of pair sign consistency and support sign concordance"
    ):
        raise RuntimeError(f"{source_label} ALE bundle is not effectively complete")

    path = response_root / "tables" / "class_feature_marker_response_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    required = {
        "stage",
        "lineage",
        "class",
        "feature",
        "feature_position",
        "mean_ale_q10_to_q90_probability_change",
        "pair_sign_consistency",
        "support_sign_concordance",
        "direction_consistency",
        "direction_consistency_threshold",
        "direction_symbol",
        "n_iterations",
        "response_metric",
        "class_normalized_importance",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{source_label} ALE summary lacks columns: {missing}")
    keys = ["stage", "lineage", "class", "feature", "feature_position"]
    if (
        len(table) != expected_rows
        or table["feature"].nunique() != EXPECTED_FEATURES
        or table[["stage", "class"]].drop_duplicates().shape[0] != expected_classes
        or table.duplicated(keys).any()
        or not table.groupby(["stage", "class"])["feature"].nunique().eq(EXPECTED_FEATURES).all()
        or not table["n_iterations"].eq(100).all()
        or not np.allclose(
            table["direction_consistency_threshold"].to_numpy(dtype=float),
            DIRECTION_CONSISTENCY_THRESHOLD,
            atol=0.0,
            rtol=0.0,
        )
        or set(table["response_metric"].astype(str))
        != {"patient-equal first-order ALE q10-to-q90 contrast"}
    ):
        raise RuntimeError(
            f"{source_label} ALE summary must contain {expected_classes} classes x "
            f"{EXPECTED_FEATURES} features x 100 iterations"
        )

    mean = table["mean_ale_q10_to_q90_probability_change"].to_numpy(dtype=float)
    sign = np.where(
        mean > DIRECTION_ZERO_TOLERANCE,
        1,
        np.where(mean < -DIRECTION_ZERO_TOLERANCE, -1, 0),
    ).astype(int)
    stable = (
        (sign != 0)
        & table["pair_sign_consistency"].to_numpy(dtype=float).__ge__(
            DIRECTION_CONSISTENCY_THRESHOLD
        )
        & table["support_sign_concordance"].to_numpy(dtype=float).__ge__(
            DIRECTION_CONSISTENCY_THRESHOLD
        )
    )
    expected_symbol = np.where(stable, np.where(sign > 0, "+", "−"), "±")
    if not np.array_equal(table["direction_symbol"].astype(str).to_numpy(), expected_symbol):
        raise RuntimeError(f"{source_label} ALE symbols do not match the saved response")
    table["ale_direction_sign"] = sign
    table["ale_direction_stable"] = stable
    table["ale_response_root"] = str(response_root)
    table["ale_manifest_path"] = str(manifest_path)
    return table.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _add_ale_directional_semantics(
    *,
    run_root: Path,
    response_root: Path,
    weights: pd.DataFrame,
    weight_column: str,
    expected_rows: int,
    expected_classes: int,
    source_label: str,
) -> pd.DataFrame:
    """Attach stable DLBCL ALE direction to pinned permutation magnitude."""

    ale = _validated_ale_summary(
        response_root,
        expected_rows=expected_rows,
        expected_classes=expected_classes,
        source_label=source_label,
    )
    joint = pd.read_csv(Path(run_root) / "tables" / "class_feature_joint_importance_summary.csv")
    if "rank_within_class" not in joint:
        raise ValueError(f"{source_label} joint importance lacks rank_within_class")
    ranks = joint[
        ["stage", "lineage", "class", "feature", "rank_within_class"]
    ].rename(columns={"rank_within_class": "specificity_rank_within_class"})
    ale_columns = [
        "stage",
        "lineage",
        "class",
        "feature",
        "feature_position",
        "mean_ale_q10_to_q90_probability_change",
        "pair_sign_consistency",
        "support_sign_concordance",
        "direction_consistency",
        "direction_consistency_threshold",
        "direction_symbol",
        "ale_direction_sign",
        "ale_direction_stable",
        "ale_response_root",
        "ale_manifest_path",
        "class_normalized_importance",
    ]
    table = weights.merge(
        ranks,
        on=["stage", "lineage", "class", "feature"],
        how="left",
        validate="one_to_one",
    ).merge(
        ale[ale_columns],
        on=["stage", "lineage", "class", "feature", "feature_position"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_ale"),
    )
    if table[["specificity_rank_within_class", "ale_direction_sign"]].isna().any().any():
        raise RuntimeError(f"{source_label} ALE direction/rank join is incomplete")
    ale_weight_column = (
        "class_normalized_importance_ale"
        if "class_normalized_importance_ale" in table.columns
        else "class_normalized_importance"
    )
    if not np.allclose(
        table[weight_column].to_numpy(dtype=float),
        table[ale_weight_column].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(f"{source_label} ALE bundle does not match pinned importance")

    ranked = table.sort_values(
        ["stage", "class", weight_column, "feature_position"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()
    ranked["magnitude_rank_within_class"] = (
        ranked.groupby(["stage", "class"], sort=False).cumcount() + 1
    )
    table = table.merge(
        ranked[["stage", "class", "feature", "magnitude_rank_within_class"]],
        on=["stage", "class", "feature"],
        how="left",
        validate="one_to_one",
    )
    table["class_importance_magnitude"] = table[weight_column]
    table["secondary_mean_direction_sign"] = table["ale_direction_sign"].astype(int)
    table["secondary_direction_consistency_fraction"] = table[
        "direction_consistency"
    ].astype(float)
    table["secondary_direction_consistency_threshold"] = DIRECTION_CONSISTENCY_THRESHOLD
    table["secondary_direction_consistent"] = table["ale_direction_stable"].astype(bool)
    table["secondary_stable_direction_sign"] = np.where(
        table["secondary_direction_consistent"],
        table["secondary_mean_direction_sign"],
        0,
    ).astype(int)
    # Compatibility aliases retained for the existing proxy-collapse/export schema.
    table["primary_mean_direction_sign"] = table["ale_direction_sign"].astype(int)
    table["primary_direction_consistency_fraction"] = table[
        "pair_sign_consistency"
    ].astype(float)
    table["primary_secondary_direction_agreement"] = table[
        "support_sign_concordance"
    ].ge(DIRECTION_CONSISTENCY_THRESHOLD)
    table["primary_secondary_agreement_sensitivity_gate"] = table[
        "ale_direction_stable"
    ].astype(bool)
    table["primary_secondary_agreement_used_for_selection"] = False
    table["direction_metric"] = "patient-equal first-order ALE q10-to-q90 contrast"
    table["direction_scope_used_for_weight_sign"] = (
        "all routed DLBCL cells; patient-equal local effects within each frozen pair"
    )
    table["direction_rows_filtered_upstream"] = False
    table["direction_sign_source"] = "DLBCL q10-to-q90 first-order ALE response"
    table["direction_aware_signed_weight_before_coverage"] = (
        table["secondary_stable_direction_sign"] * table[weight_column]
    )
    table["rank_within_class_semantics"] = "specificity_rank_not_magnitude_rank"
    expected_ranks = set(range(1, EXPECTED_FEATURES + 1))
    for _, frame in table.groupby(["stage", "class"], sort=False):
        if (
            set(frame["specificity_rank_within_class"].astype(int)) != expected_ranks
            or set(frame["magnitude_rank_within_class"].astype(int)) != expected_ranks
        ):
            raise RuntimeError(f"{source_label} does not expose complete explicit ranks")
    return table


def _load_manifest(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "manifests" / "run_manifest.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("input_unchanged") is not True:
        return None
    if payload.get("execution_profile") != "fast_chunk":
        return None
    if payload.get("selected_models") != ["XGB", "MLP"]:
        return None
    if payload.get("selected_iterations") != list(range(1, 101)):
        return None
    if int(payload.get("feature_count", -1)) != EXPECTED_FEATURES:
        return None
    if int(payload.get("subtype_count", -1)) != EXPECTED_CLASSES:
        return None
    return payload


def _validate_joint_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    required = {
        "stage",
        "lineage",
        "class",
        "feature",
        "signed_probability_drop_in_baseline_class",
        "class_normalized_importance",
        "positive_effect_fraction",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Step04B joint summary lacks columns: {missing}")
    if (
        len(table) != EXPECTED_FEATURE_ROWS
        or table["feature"].nunique() != EXPECTED_FEATURES
        or table[["stage", "class"]].drop_duplicates().shape[0] != EXPECTED_CLASSES
    ):
        raise RuntimeError("Step04B joint summary must contain 13 classes x 50 features")
    return table


def _validate_paired_tables(run_root: Path) -> pd.DataFrame:
    frequency_path = run_root / "tables" / "paired_iteration_top20_attribution_frequency.csv"
    reconciliation_path = run_root / "tables" / "paired_iteration_vs_ensemble_reconciliation.csv"
    if not frequency_path.is_file() or not reconciliation_path.is_file():
        raise FileNotFoundError("Paired attribution tables are incomplete")
    frequency = pd.read_csv(frequency_path)
    required = {
        "stage",
        "lineage",
        "class",
        "feature",
        "feature_position",
        "positive_top_k_count",
        "mean_iteration_signed_probability_drop",
        "positive_top_k_attribution_frequency",
        "n_iterations",
        "top_k",
    }
    missing = sorted(required - set(frequency.columns))
    if missing:
        raise ValueError(f"Paired attribution table lacks columns: {missing}")
    if (
        len(frequency) != EXPECTED_FEATURE_ROWS
        or not frequency["n_iterations"].eq(100).all()
        or not frequency["top_k"].eq(20).all()
        or frequency.duplicated(["stage", "class", "feature"]).any()
        or not frequency["positive_top_k_count"].between(0, 100).all()
        or not np.allclose(
            frequency["positive_top_k_attribution_frequency"].to_numpy(dtype=float),
            frequency["positive_top_k_count"].to_numpy(dtype=float) / 100.0,
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise RuntimeError("Paired attribution contract must be 13 x 50, 100 iterations, top-20")
    reconciliation = pd.read_csv(reconciliation_path)
    if len(reconciliation) != EXPECTED_FEATURE_ROWS or not _as_bool(
        reconciliation["within_tolerance"]
    ).all():
        raise RuntimeError("Paired attribution does not reconcile with ensemble importance")
    return frequency


def _validated_run(run_root: Path) -> Step04BRun | None:
    """Return one validated completed run, or ``None`` when its contract fails."""

    manifest = _load_manifest(run_root)
    if manifest is None:
        return None
    completed_at = str(manifest.get("completed_at", ""))
    paired = manifest.get("paired_iteration_attribution", {})
    if paired.get("enabled") is True and int(paired.get("top_k", -1)) == 20:
        try:
            _validate_joint_summary(
                run_root / "tables" / "class_feature_joint_importance_summary.csv"
            )
            _validate_paired_tables(run_root)
        except (FileNotFoundError, ValueError, RuntimeError):
            return None
        return Step04BRun(
            run_root.name, run_root, manifest, "paired_iteration", completed_at
        )
    try:
        _validate_joint_summary(
            run_root / "tables" / "class_feature_joint_importance_summary.csv"
        )
    except (FileNotFoundError, ValueError, RuntimeError):
        return None
    return Step04BRun(run_root.name, run_root, manifest, "joint_ensemble", completed_at)


def resolve_step04b_run(run_id: str | None = None) -> Step04BRun:
    """Resolve one explicitly pinned immutable completed Step04B run."""

    if run_id is None:
        raise ValueError("An explicit pinned Step04B run ID is required")
    if RUN_NAME.fullmatch(run_id) is None:
        raise ValueError(f"Unsafe Step04B run ID: {run_id!r}")
    run_root = (RUNS_ROOT / run_id).resolve()
    if run_root.parent != RUNS_ROOT.resolve() or not run_root.is_dir():
        raise FileNotFoundError(f"Pinned Step04B run is missing: {run_root}")
    selected = _validated_run(run_root)
    if selected is None:
        raise RuntimeError(f"Pinned Step04B run does not satisfy the completed contract: {run_id}")
    return selected


def load_step04b_class_weights(
    run_id: str | None = None,
    *,
    require_paired: bool = False,
) -> tuple[Step04BRun, pd.DataFrame]:
    """Return the completed ensemble weight and optional paired stability metadata."""

    selected = resolve_step04b_run(run_id)
    if require_paired and selected.method != "paired_iteration":
        raise RuntimeError(
            "This scRNA signature contract requires a completed paired-attribution run"
        )
    joint = _validate_joint_summary(
        selected.root / "tables" / "class_feature_joint_importance_summary.csv"
    )
    table = joint[
        [
            "stage",
            "lineage",
            "class",
            "feature",
            "signed_probability_drop_in_baseline_class",
            "class_normalized_importance",
        ]
    ].rename(
        columns={
            "signed_probability_drop_in_baseline_class": "signed_attribution",
            "class_normalized_importance": "step04b_class_weight",
        }
    )
    if selected.method == "paired_iteration":
        frequency = _validate_paired_tables(selected.root)[
            [
                "stage",
                "lineage",
                "class",
                "feature",
                "feature_position",
                "positive_top_k_count",
                "mean_iteration_signed_probability_drop",
                "positive_top_k_attribution_frequency",
                "n_iterations",
                "top_k",
            ]
        ].rename(
            columns={
                "positive_top_k_attribution_frequency": "stability_fraction",
            }
        )
        table = table.merge(
            frequency,
            on=["stage", "lineage", "class", "feature"],
            how="left",
            validate="one_to_one",
        )
        stability_metric = "positive top-20 attribution frequency across paired iterations"
        paired_available = True
    else:
        table["stability_fraction"] = math.nan
        stability_metric = "not available until a completed paired-iteration run exists"
        paired_available = False
    table = table.copy()
    if bool(table["step04b_class_weight"].lt(0).any()):
        raise RuntimeError("Step04B class-normalized importance contains a negative weight")
    totals = table.groupby(["stage", "class"])["step04b_class_weight"].transform("sum")
    if not np.allclose(totals.to_numpy(dtype=float), 1.0, atol=1e-10, rtol=0.0):
        raise RuntimeError("Step04B class-normalized importance does not sum to one per class")
    table["stability_metric"] = stability_metric
    table["paired_stability_available"] = paired_available
    table["source_method"] = selected.method
    table["source_run_id"] = selected.run_id
    table["source_completed_at"] = selected.completed_at
    table["weight_definition"] = (
        "class_normalized_importance from the same completed joint XGB+MLP run"
    )
    table["frequency_used_as_weight"] = False
    return selected, table.sort_values(
        ["stage", "class", "step04b_class_weight", "feature"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)



def load_ale_directional_step04b_class_weights(
    run_id: str,
) -> tuple[Step04BRun, pd.DataFrame]:
    """Load pinned Step04B importance with the completed DLBCL ALE sign."""

    selected, weights = load_step04b_class_weights(run_id, require_paired=True)
    table = _add_ale_directional_semantics(
        run_root=selected.root,
        response_root=STEP04B_ALE_ROOT,
        weights=weights,
        weight_column="step04b_class_weight",
        expected_rows=EXPECTED_FEATURE_ROWS,
        expected_classes=EXPECTED_CLASSES,
        source_label="Step04B",
    )
    return selected, table


def _validate_step02_manifest(run_root: Path) -> dict[str, Any]:
    path = run_root / "manifests" / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_contract = (
        payload.get("status") == "complete"
        and payload.get("input_unchanged") is True
        and payload.get("execution_profile") == "fast_chunk"
        and payload.get("retraining") is False
        and payload.get("source_models_read_only") is True
        and payload.get("source_h5ad_read_only") is True
        and payload.get("selected_models") == ["XGB", "MLP"]
        and payload.get("selected_iterations") == list(range(1, 101))
        and int(payload.get("feature_count", -1)) == EXPECTED_FEATURES
        and int(payload.get("class_count", -1)) == EXPECTED_BROAD_CLASSES
        and float(payload.get("baseline_saved_prediction_agreement", -1.0)) == 1.0
    )
    paired = payload.get("paired_iteration_attribution", {})
    if not required_contract or paired.get("enabled") is not True or int(
        paired.get("top_k", -1)
    ) != 20:
        raise RuntimeError(
            f"Pinned Step02 run does not satisfy the completed 5-class paired contract: {path}"
        )
    return payload


def _validate_step02_joint_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    required = {
        "stage",
        "lineage",
        "class",
        "feature",
        "signed_probability_drop_in_baseline_class",
        "class_normalized_importance",
        "positive_effect_fraction",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Step02 broad joint summary lacks columns: {missing}")
    class_feature = table[["class", "feature"]]
    if (
        len(table) != EXPECTED_BROAD_FEATURE_ROWS
        or table["feature"].nunique() != EXPECTED_FEATURES
        or table[["stage", "class"]].drop_duplicates().shape[0]
        != EXPECTED_BROAD_CLASSES
        or set(table["stage"].astype(str)) != {"broad"}
        or set(table["lineage"].astype(str)) != {"Broad lineage"}
        or set(table["class"].astype(str)) != set(BROAD_CLASS_ORDER)
        or class_feature.duplicated().any()
        or not table.groupby("class")["feature"].nunique().eq(EXPECTED_FEATURES).all()
    ):
        raise RuntimeError("Step02 broad summary must contain 5 classes x 50 features")
    signed = table["signed_probability_drop_in_baseline_class"].to_numpy(dtype=float)
    normalized = table["class_normalized_importance"].to_numpy(dtype=float)
    if np.any(normalized < 0) or not np.array_equal(normalized > 0, signed > 0):
        raise RuntimeError(
            "Step02 class_normalized_importance must be positive exactly for positive attribution"
        )
    totals = table.groupby("class")["class_normalized_importance"].sum().to_numpy(
        dtype=float
    )
    if not np.allclose(totals, 1.0, atol=1e-10, rtol=0.0):
        raise RuntimeError("Step02 broad class-normalized importance does not sum to one")
    return table


def _validate_step02_paired_tables(run_root: Path) -> pd.DataFrame:
    frequency_path = run_root / "tables" / "paired_iteration_top20_attribution_frequency.csv"
    reconciliation_path = (
        run_root / "tables" / "paired_iteration_vs_ensemble_reconciliation.csv"
    )
    if not frequency_path.is_file() or not reconciliation_path.is_file():
        raise FileNotFoundError("Step02 broad paired attribution tables are incomplete")
    frequency = pd.read_csv(frequency_path)
    required = {
        "stage",
        "lineage",
        "class",
        "feature",
        "feature_position",
        "positive_top_k_count",
        "mean_iteration_signed_probability_drop",
        "positive_top_k_attribution_frequency",
        "n_iterations",
        "top_k",
    }
    missing = sorted(required - set(frequency.columns))
    if missing:
        raise ValueError(f"Step02 broad paired table lacks columns: {missing}")
    if (
        len(frequency) != EXPECTED_BROAD_FEATURE_ROWS
        or frequency.duplicated(["stage", "class", "feature"]).any()
        or not frequency["n_iterations"].eq(100).all()
        or not frequency["top_k"].eq(20).all()
        or not frequency["positive_top_k_count"].between(0, 100).all()
        or not np.allclose(
            frequency["positive_top_k_attribution_frequency"].to_numpy(dtype=float),
            frequency["positive_top_k_count"].to_numpy(dtype=float) / 100.0,
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise RuntimeError(
            "Step02 broad paired contract must be 5 x 50, 100 iterations, top-20"
        )
    reconciliation = pd.read_csv(reconciliation_path)
    if (
        len(reconciliation) != EXPECTED_BROAD_FEATURE_ROWS
        or "within_tolerance" not in reconciliation
        or not _as_bool(reconciliation["within_tolerance"]).all()
    ):
        raise RuntimeError("Step02 broad paired attribution does not reconcile")
    return frequency


def resolve_step02_broad_run(run_id: str) -> Step02Run:
    """Resolve one explicit immutable completed Step02 broad run; no latest-run fallback."""

    if RUN_NAME.fullmatch(run_id) is None:
        raise ValueError(f"Unsafe Step02 broad run ID: {run_id!r}")
    root = (STEP02_RUNS_ROOT / run_id).resolve()
    if root.parent != STEP02_RUNS_ROOT.resolve() or not root.is_dir():
        raise FileNotFoundError(f"Pinned Step02 broad run is missing: {root}")
    manifest = _validate_step02_manifest(root)
    _validate_step02_joint_summary(
        root / "tables" / "class_feature_joint_importance_summary.csv"
    )
    _validate_step02_paired_tables(root)
    return Step02Run(
        run_id=run_id,
        root=root,
        manifest=manifest,
        completed_at=str(manifest.get("completed_at", "")),
    )


def load_step02_broad_class_weights(run_id: str) -> tuple[Step02Run, pd.DataFrame]:
    """Load positive Step02 broad weights and paired top-20 stability metadata."""

    selected = resolve_step02_broad_run(run_id)
    joint = _validate_step02_joint_summary(
        selected.root / "tables" / "class_feature_joint_importance_summary.csv"
    )
    frequency = _validate_step02_paired_tables(selected.root)
    weights = joint[
        [
            "stage",
            "lineage",
            "class",
            "feature",
            "signed_probability_drop_in_baseline_class",
            "class_normalized_importance",
        ]
    ].rename(
        columns={
            "signed_probability_drop_in_baseline_class": "signed_attribution",
        }
    )
    weights = weights.merge(
        frequency[
            [
                "stage",
                "lineage",
                "class",
                "feature",
                "feature_position",
                "positive_top_k_count",
                "mean_iteration_signed_probability_drop",
                "positive_top_k_attribution_frequency",
                "n_iterations",
                "top_k",
            ]
        ].rename(
            columns={"positive_top_k_attribution_frequency": "stability_fraction"}
        ),
        on=["stage", "lineage", "class", "feature"],
        how="left",
        validate="one_to_one",
    )
    weights["source_method"] = "paired_iteration"
    weights["source_run_id"] = selected.run_id
    weights["source_completed_at"] = selected.completed_at
    weights["frequency_used_as_weight"] = False
    return selected, weights



def load_ale_directional_step02_broad_class_weights(
    run_id: str,
) -> tuple[Step02Run, pd.DataFrame]:
    """Load pinned Step02 importance with the completed DLBCL ALE sign."""

    selected, weights = load_step02_broad_class_weights(run_id)
    table = _add_ale_directional_semantics(
        run_root=selected.root,
        response_root=STEP02_ALE_ROOT,
        weights=weights,
        weight_column="class_normalized_importance",
        expected_rows=EXPECTED_BROAD_FEATURE_ROWS,
        expected_classes=EXPECTED_BROAD_CLASSES,
        source_label="Step02 broad",
    )
    return selected, table


def build_step02_broad_directional_signature_support(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Use stable DLBCL ALE direction with permutation-importance magnitude."""

    support = weights.loc[
        weights["secondary_direction_consistent"]
        & weights["class_importance_magnitude"].gt(0)
    ].copy()
    if support[["stage", "class"]].drop_duplicates().shape[0] != EXPECTED_BROAD_CLASSES:
        raise RuntimeError("Every Step02 broad class must retain stable directional support")
    support["task"] = "broad"
    support["source_stage"] = "Step02"
    support["source_class"] = support["class"].astype(str)
    direction_metrics = set(support["direction_metric"].astype(str))
    if direction_metrics != {"patient-equal first-order ALE q10-to-q90 contrast"}:
        raise RuntimeError("Step02 signature support is not DLBCL-ALE based")
    support["signature_support"] = (
        "Step02 DLBCL q10-to-q90 ALE direction with pair/support consistency >=0.8; "
        "class_normalized_importance magnitude"
    )
    support["positive_top20_frequency_role"] = "metadata_only"
    support["frequency_used_as_weight"] = False
    support["primary_secondary_agreement_used_for_selection"] = False
    return support.reset_index(drop=True)


def build_step04b_tcell_directional_signature_support(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict stable DLBCL-ALE direction to routed T-cell classes."""

    support = weights.loc[
        weights["stage"].eq("T_cells")
        & weights["secondary_direction_consistent"]
        & weights["class_importance_magnitude"].gt(0)
    ].copy()
    support["source_class"] = support["class"].astype(str)
    support["class"] = support["source_class"].map(TCELL_CLASS_DISPLAY)
    if support["class"].isna().any() or set(support["class"].astype(str)) != set(
        TCELL_CLASS_DISPLAY.values()
    ):
        raise RuntimeError("Every routed Step04B T-cell class must retain stable support")
    support["task"] = "tcell_subtype"
    support["source_stage"] = "Step04B"
    support["class_normalized_importance"] = support["step04b_class_weight"]
    direction_metrics = set(support["direction_metric"].astype(str))
    if direction_metrics != {"patient-equal first-order ALE q10-to-q90 contrast"}:
        raise RuntimeError("Step04B signature support is not DLBCL-ALE based")
    support["signature_support"] = (
        "Step04B T-cell DLBCL q10-to-q90 ALE direction with pair/support "
        "consistency >=0.8; class_normalized_importance magnitude"
    )
    support["positive_top20_frequency_role"] = "metadata_only"
    support["frequency_used_as_weight"] = False
    support["primary_secondary_agreement_used_for_selection"] = False
    return support.reset_index(drop=True)


def collapse_directional_signature_support_to_proxy_groups(
    signature_support: pd.DataFrame,
    coverage_path: Path,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse stable support and L1-normalize signed weights after coverage."""

    required_support = {
        "task",
        "source_stage",
        "lineage",
        "class",
        "source_class",
        "feature",
        "class_importance_magnitude",
        "direction_aware_signed_weight_before_coverage",
        "secondary_mean_direction_sign",
        "secondary_stable_direction_sign",
        "secondary_direction_consistency_fraction",
        "secondary_direction_consistent",
        "primary_secondary_direction_agreement",
        "primary_secondary_agreement_sensitivity_gate",
        "specificity_rank_within_class",
        "magnitude_rank_within_class",
        "stability_fraction",
        "source_method",
        "source_run_id",
        "source_completed_at",
    }
    missing_support = sorted(required_support - set(signature_support.columns))
    if missing_support:
        raise ValueError(f"Directional signature support lacks columns: {missing_support}")
    if not _as_bool(signature_support["secondary_direction_consistent"]).all():
        raise RuntimeError("Directional support contains a feature below the consistency gate")

    coverage = pd.read_csv(coverage_path)
    required_coverage = {"codex_marker", "feature_col", "covered", "present_genes"}
    missing_coverage = sorted(required_coverage - set(coverage.columns))
    if missing_coverage:
        raise ValueError(f"Coverage table lacks columns: {missing_coverage}")
    coverage = coverage.copy()
    coverage["included"] = _as_bool(coverage["covered"])
    coverage["proxy_group"] = coverage["present_genes"].fillna("").astype(str).map(
        lambda value: "|".join(sorted(filter(None, value.split("|"))))
    )
    coverage["included"] &= coverage["proxy_group"].ne("")
    merged = signature_support.merge(
        coverage[
            ["codex_marker", "feature_col", "included", "proxy_group", "present_genes"]
        ],
        left_on="feature",
        right_on="codex_marker",
        how="left",
        validate="many_to_one",
    )
    merged["included"] = merged["included"].fillna(False).astype(bool)
    merged["dataset"] = dataset
    included = merged.loc[merged["included"]].copy()
    if included.empty:
        raise RuntimeError(f"No covered stable directional support for {dataset}")

    proxy = (
        included.groupby(
            ["dataset", "task", "source_stage", "lineage", "class", "proxy_group"],
            as_index=False,
        )
        .agg(
            markers=("feature", lambda values: "|".join(sorted(set(map(str, values))))),
            feature_cols=(
                "feature_col",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            direction_aware_proxy_weight_raw=(
                "direction_aware_signed_weight_before_coverage",
                "sum",
            ),
            magnitude_mass_before_alias_collapse=("class_importance_magnitude", "sum"),
            min_secondary_direction_consistency_fraction=(
                "secondary_direction_consistency_fraction",
                "min",
            ),
            max_secondary_direction_consistency_fraction=(
                "secondary_direction_consistency_fraction",
                "max",
            ),
            all_primary_secondary_direction_agree=(
                "primary_secondary_direction_agreement",
                "all",
            ),
            all_primary_secondary_sensitivity_gate=(
                "primary_secondary_agreement_sensitivity_gate",
                "all",
            ),
            mean_positive_top20_frequency=("stability_fraction", "mean"),
            source_run_id=("source_run_id", "first"),
            source_completed_at=("source_completed_at", "first"),
        )
    )
    proxy["equal_weight_proxy_weight_raw"] = 1.0
    keys = ["task", "class"]
    equal_totals = proxy.groupby(keys)["equal_weight_proxy_weight_raw"].transform("sum")
    signed_l1_totals = proxy.groupby(keys)["direction_aware_proxy_weight_raw"].transform(
        lambda values: float(np.abs(values.to_numpy(dtype=float)).sum())
    )
    if bool(equal_totals.le(0).any() or signed_l1_totals.le(0).any()):
        raise RuntimeError(f"Directional proxy normalization failed for {dataset}")
    proxy["equal_weight_proxy_weight"] = proxy["equal_weight_proxy_weight_raw"] / equal_totals
    proxy["direction_aware_proxy_weight"] = (
        proxy["direction_aware_proxy_weight_raw"] / signed_l1_totals
    )
    proxy["direction_aware_proxy_weight_sign"] = np.sign(
        proxy["direction_aware_proxy_weight"]
    ).astype(int)
    proxy["weight_comparison_contract"] = (
        "identical stable unique proxy support; equal positive weight versus signed "
        "class_normalized_importance normalized by sum(abs(weight)) after coverage"
    )
    proxy["positive_top20_frequency_role"] = "metadata_only"
    proxy["frequency_used_as_weight"] = False
    proxy["primary_secondary_agreement_used_for_selection"] = False
    equal_sums = proxy.groupby(keys)["equal_weight_proxy_weight"].sum().to_numpy(dtype=float)
    signed_l1_sums = proxy.groupby(keys)["direction_aware_proxy_weight"].apply(
        lambda values: float(np.abs(values.to_numpy(dtype=float)).sum())
    ).to_numpy(dtype=float)
    if not np.allclose(equal_sums, 1.0, atol=1e-10, rtol=0.0):
        raise RuntimeError(f"Equal proxy weights do not sum to one for {dataset}")
    if not np.allclose(signed_l1_sums, 1.0, atol=1e-10, rtol=0.0):
        raise RuntimeError(f"Signed proxy weights do not have unit L1 norm for {dataset}")

    audit_columns = [
        "dataset",
        "task",
        "source_stage",
        "lineage",
        "class",
        "source_class",
        "feature",
        "class_importance_magnitude",
        "direction_aware_signed_weight_before_coverage",
        "secondary_mean_direction_sign",
        "secondary_stable_direction_sign",
        "secondary_direction_consistency_fraction",
        "secondary_direction_consistent",
        "primary_secondary_direction_agreement",
        "primary_secondary_agreement_sensitivity_gate",
        "primary_secondary_agreement_used_for_selection",
        "specificity_rank_within_class",
        "magnitude_rank_within_class",
        "rank_within_class_semantics",
        "stability_fraction",
        "included",
        "proxy_group",
        "feature_col",
        "source_method",
        "source_run_id",
    ]
    return (
        proxy.sort_values(
            ["task", "class", "direction_aware_proxy_weight", "proxy_group"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True),
        merged[audit_columns]
        .sort_values(["task", "class", "magnitude_rank_within_class", "feature"])
        .reset_index(drop=True),
    )
