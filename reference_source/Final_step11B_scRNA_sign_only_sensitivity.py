#!/usr/bin/env python3
# %% [markdown]
# # Final Step 11B. scRNA marker/sign/importance sensitivity
#
# Compare three fixed scoring rules on identical scRNA proxy support without
# changing the canonical Step11 bundle or the Step12 UMAP panels.

# %% [markdown]
# ## 11B-0. Execution contract
#
# The primary compatibility ablation uses the canonical post-alias-collapse
# sign. A strict sensitivity removes the two mixed-direction PTPRC proxy-class
# rows from all three arms before renormalization. Outputs are written only to
# a new sensitivity run and a matching noncanonical Figure reexport directory.

# %%
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import final_fixed_signature_validation as fsv


FINAL_ROOT = Path("__ROIDER_ROOT__/Analysis/python/Final")
SOURCE_RUN_ROOT = fsv.COMPLETED_BUNDLE_ROOT
SOURCE_TABLES = SOURCE_RUN_ROOT / "tables"
SENSITIVITY_ROOT = FINAL_ROOT / "tables" / "scRNA" / "fixed_signature" / "sensitivity"
FIGURE_ROOT = FINAL_ROOT / "figures" / "v2" / "scRNA" / "reexports"
DEFAULT_RUN_ID = "sign_only_equal_magnitude_20260824_172147"
DEFAULT_ARIAL_REGULAR = Path("__FONT_ROOT__/arial.ttf")
DEFAULT_ARIAL_BOLD = Path("__FONT_ROOT__/arialbd.ttf")

CONTRACTS = ("canonical_compatible", "strict_conflict_free")
METHODS = {
    "marker_only": {
        "display": "Marker only",
        "weight_col": "ablation_marker_weight",
    },
    "collapsed_sign_only": {
        "display": "ALE sign only",
        "weight_col": "ablation_sign_weight",
    },
    "sign_plus_importance": {
        "display": "ALE sign × PFI weight",
        "weight_col": "ablation_sign_importance_weight",
    },
}
FIGURE_STEMS = {
    "GSE182434": "Final_mv1_Step11B-01_GSE182434_marker_sign_importance_ablation_performance",
    "Roider": "Final_mv1_Step11B-02_Roider_marker_sign_importance_ablation_performance",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_new_run_id(run_id: str) -> None:
    if fsv.RUN_NAME.fullmatch(run_id or "") is None:
        raise ValueError(f"Unsafe sensitivity run ID: {run_id!r}")
    for root in (SENSITIVITY_ROOT / run_id, FIGURE_ROOT / run_id):
        if root.exists() or root.with_name(root.name + ".partial").exists():
            raise FileExistsError(f"Sensitivity output already exists: {root}")


def _load_component_directions() -> pd.DataFrame:
    path = SOURCE_TABLES / "combined_fixed_class_direction_signature_features.csv"
    table = pd.read_csv(path)
    required = {
        "task",
        "class",
        "feature",
        "ale_direction_sign",
        "class_importance_magnitude",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Component-direction table lacks columns: {missing}")
    keys = ["task", "class", "feature"]
    if bool(table.duplicated(keys).any()):
        raise RuntimeError("Component-direction keys are not unique")
    table["ale_direction_sign"] = pd.to_numeric(
        table["ale_direction_sign"], errors="raise"
    ).astype(int)
    if not table["ale_direction_sign"].isin([-1, 1]).all():
        raise RuntimeError("Component directions must be exactly -1 or +1")
    return table


def _prepare_proxy_weights(
    dataset: str,
    component_directions: pd.DataFrame,
    *,
    contract: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if contract not in CONTRACTS:
        raise KeyError(contract)
    weights = pd.read_csv(SOURCE_TABLES / f"{dataset}_signature_proxy_weights.csv")
    component_lookup = {
        (str(row["task"]), str(row["class"]), str(row["feature"])): (
            int(row["ale_direction_sign"]),
            float(row["class_importance_magnitude"]),
        )
        for _, row in component_directions.iterrows()
    }
    component_signs: list[str] = []
    component_importances: list[str] = []
    alias_conflicts: list[bool] = []
    unweighted_signs: list[int] = []
    for _, row in weights.iterrows():
        markers = str(row["markers"]).split("|")
        components = [
            component_lookup[(str(row["task"]), str(row["class"]), marker)]
            for marker in markers
        ]
        signs = [value[0] for value in components]
        component_signs.append("|".join(map(str, signs)))
        component_importances.append("|".join(f"{value[1]:.17g}" for value in components))
        alias_conflicts.append(len(set(signs)) > 1)
        unweighted_signs.append(int(np.sign(sum(signs))))
    weights["component_ale_direction_signs"] = component_signs
    weights["component_class_importance_magnitudes"] = component_importances
    weights["alias_component_direction_conflict"] = alias_conflicts
    weights["component_unweighted_sign"] = unweighted_signs
    audit = weights.loc[weights["alias_component_direction_conflict"]].copy()
    if contract == "strict_conflict_free":
        weights = weights.loc[~weights["alias_component_direction_conflict"]].copy()
        weights["ablation_sign_raw"] = weights["component_unweighted_sign"].astype(float)
    else:
        weights["ablation_sign_raw"] = pd.to_numeric(
            weights["direction_aware_proxy_weight_sign"], errors="raise"
        ).astype(float)
    if bool(weights["ablation_sign_raw"].eq(0).any()):
        raise RuntimeError(f"Zero sign remains under {contract}/{dataset}")
    keys = ["task", "class"]
    weights["ablation_marker_weight"] = 1.0 / weights.groupby(keys)[
        "proxy_group"
    ].transform("size")
    sign_l1 = weights.groupby(keys)["ablation_sign_raw"].transform(
        lambda values: float(np.abs(values.to_numpy(dtype=float)).sum())
    )
    full_raw = pd.to_numeric(
        weights["direction_aware_proxy_weight_raw"], errors="raise"
    ).astype(float)
    full_l1 = full_raw.abs().groupby([weights["task"], weights["class"]]).transform("sum")
    if bool(sign_l1.le(0).any() or full_l1.le(0).any()):
        raise RuntimeError(f"Invalid L1 normalization under {contract}/{dataset}")
    weights["ablation_sign_weight"] = weights["ablation_sign_raw"] / sign_l1
    weights["ablation_sign_importance_weight"] = full_raw / full_l1
    weights["ablation_contract"] = contract
    weights["sign_only_definition"] = (
        "equal absolute magnitude after alias collapse; class-wise signed L1 normalization"
        if contract == "canonical_compatible"
        else "mixed-direction aliases removed before equal-magnitude signed L1 normalization"
    )
    grouped = weights.groupby(keys, sort=False)
    checks = pd.DataFrame(
        {
            "marker_sum": grouped["ablation_marker_weight"].sum(),
            "sign_l1": grouped["ablation_sign_weight"].apply(
                lambda values: float(np.abs(values.to_numpy(dtype=float)).sum())
            ),
            "full_l1": grouped["ablation_sign_importance_weight"].apply(
                lambda values: float(np.abs(values.to_numpy(dtype=float)).sum())
            ),
        }
    )
    if not np.allclose(checks.to_numpy(dtype=float), 1.0, atol=1e-10, rtol=0.0):
        raise RuntimeError(f"Weight normalization failed under {contract}/{dataset}")
    audit["dataset"] = dataset
    return weights.reset_index(drop=True), audit.reset_index(drop=True)


def _load_obs_without_umap(dataset: str) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(fsv.DATASETS[dataset]["h5ad"])
    before = path.stat()
    backed = ad.read_h5ad(path, backed="r")
    try:
        obs = backed.obs.copy()
    finally:
        backed.file.close()
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"Read-only proxy H5AD changed while reading: {path}")
    obs.index = obs.index.astype(str)
    return obs, {
        "path": str(path),
        "size_bytes": int(after.st_size),
        "mtime_ns_before": int(before.st_mtime_ns),
        "mtime_ns_after": int(after.st_mtime_ns),
        "input_unchanged": True,
        "umap_read": False,
    }


def _score_three_arms(obs: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    zscores = fsv._proxy_zscores(obs, weights)
    predictions = pd.DataFrame(index=obs.index)
    for task, class_order in fsv.TASK_CLASS_ORDER.items():
        task_weights = weights.loc[weights["task"].eq(task)]
        if set(task_weights["class"].astype(str)) != set(class_order):
            raise RuntimeError(f"Class support mismatch for {task}")
        for method, spec in METHODS.items():
            class_scores: list[np.ndarray] = []
            for class_name in class_order:
                score = np.zeros(len(obs), dtype=float)
                class_rows = task_weights.loc[task_weights["class"].eq(class_name)]
                for row in class_rows.itertuples(index=False):
                    score += zscores[str(row.proxy_group)] * float(
                        getattr(row, str(spec["weight_col"]))
                    )
                class_scores.append(score)
            matrix = np.column_stack(class_scores)
            prediction = np.asarray(class_order, dtype=object)[np.argmax(matrix, axis=1)]
            sorted_scores = np.sort(matrix, axis=1)
            margin = sorted_scores[:, -1] - sorted_scores[:, -2]
            prediction[np.isclose(margin, 0.0, atol=1e-12, rtol=0.0)] = "Unassigned"
            predictions[f"pred_{task}_{method}"] = prediction
            predictions[f"score_margin_{task}_{method}"] = margin
    predictions.index.name = "cell_id"
    return predictions.reset_index()


def _evaluate_three_arms(
    dataset: str,
    obs: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    contract: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    predictions = predictions.set_index("cell_id")
    pooled_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    confusion_tables: dict[str, pd.DataFrame] = {}
    report_tables: dict[str, pd.DataFrame] = {}
    group_col = str(fsv.DATASETS[dataset]["group_col"])
    if group_col not in obs.columns:
        raise KeyError(f"Grouping column missing for {dataset}: {group_col}")
    for task in fsv._make_tasks(dataset, obs):
        mask = task.truth.isin(task.reference_labels)
        truth = task.truth.loc[mask].astype(str)
        mask_hash = fsv._evaluation_mask_sha256(obs.index[mask])
        semantic_hash = fsv._evaluation_semantic_sha256(
            dataset, task.task_id, obs.index, task.truth, mask
        )
        for method in METHODS:
            pred = fsv._display_predictions(
                predictions.loc[mask, f"pred_{task.task_id}_{method}"]
            )
            metrics = fsv._evaluate_vectors(
                truth, pred, task.reference_labels, task.model_labels
            )
            pooled_rows.append(
                {
                    "dataset": dataset,
                    "task": task.task_id,
                    "method": method,
                    "method_display": METHODS[method]["display"],
                    "ablation_contract": contract,
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
            confusion = pd.crosstab(truth, pred).reindex(
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
                    truth,
                    pred,
                    labels=[label for label in task.reference_labels if label in set(truth)],
                    output_dict=True,
                    zero_division=0,
                )
            ).T
            groups = obs.loc[mask, group_col].astype("string").fillna("Missing")
            grouped = pd.DataFrame(
                {"truth": truth, "prediction": pred, "group": groups}, index=truth.index
            )
            for group_name, frame in grouped.groupby("group", sort=True):
                group_metrics = fsv._evaluate_vectors(
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
                        "ablation_contract": contract,
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


def _pairwise_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("marker_to_sign", "marker_only", "collapsed_sign_only"),
        ("sign_to_importance", "collapsed_sign_only", "sign_plus_importance"),
    ]
    value_columns = [
        "balanced_accuracy",
        "macro_f1_reference_present_classes",
        "fraction_pred_outside_reference_label_space",
    ]
    for (contract, dataset, task), frame in metrics.groupby(
        ["ablation_contract", "dataset", "task"], sort=False
    ):
        indexed = frame.set_index("method")
        for comparison, baseline, enhanced in comparisons:
            for metric in value_columns:
                rows.append(
                    {
                        "ablation_contract": contract,
                        "dataset": dataset,
                        "task": task,
                        "comparison": comparison,
                        "baseline_method": baseline,
                        "enhanced_method": enhanced,
                        "metric": metric,
                        "baseline_value": float(indexed.loc[baseline, metric]),
                        "enhanced_value": float(indexed.loc[enhanced, metric]),
                        "delta": float(
                            indexed.loc[enhanced, metric] - indexed.loc[baseline, metric]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _save_performance_figure(
    dataset: str,
    metrics: pd.DataFrame,
    figure_root: Path,
) -> tuple[Path, Path]:
    from final_fixed_signature_validation import _save_review_figure_pair

    methods = list(METHODS)
    labels = ["Marker\nonly", "ALE sign\nonly", "ALE sign ×\nPFI weight"]
    frame_all = metrics.loc[
        metrics["ablation_contract"].eq("canonical_compatible")
        & metrics["dataset"].eq(dataset)
    ]
    fig, axes = plt.subplots(2, 1, figsize=(5.15, 6.9), facecolor="white")
    task_frames: dict[str, pd.DataFrame] = {}
    for ax, task, task_title in zip(
        axes,
        ["broad", "tcell_subtype"],
        ["Broad lineage", "T-cell subtype"],
    ):
        frame = frame_all.loc[frame_all["task"].eq(task)].set_index("method").reindex(methods)
        if frame.isna().any().any():
            raise RuntimeError(f"Incomplete Figure metrics for {dataset}/{task}")
        task_frames[task] = frame
        n_values = frame["n_cells"].astype(int).unique()
        if len(n_values) != 1:
            raise RuntimeError(f"Inconsistent denominator for {dataset}/{task}")
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
                    fontsize=7.8,
                )
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title(
            f"{task_title} vs scRNA reference\n(n={int(n_values[0]):,} cells)",
            pad=8,
            fontsize=10.8,
        )
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        f"{dataset}: CODEX ML-derived marker-signature ablation",
        y=0.992,
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.922,
        "Fixed proxy support; no scRNA fitting or paired same-cell comparison",
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
    subtype = task_frames["tcell_subtype"][
        "fraction_pred_outside_reference_label_space"
    ].astype(float)
    footer = (
        "Subtype outside-label calls retained as errors: "
        f"marker {subtype['marker_only']:.1%}; sign {subtype['collapsed_sign_only']:.1%}; "
        f"sign × PFI {subtype['sign_plus_importance']:.1%}.\n"
        "Post-collapse compatibility ablation; 2 mixed-direction PTPRC proxy-class rows. "
        "Pooled-cell descriptive metrics."
    )
    fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=6.5, linespacing=1.2)
    fig.subplots_adjust(left=0.15, right=0.975, top=0.79, bottom=0.19, hspace=0.48)
    try:
        png, pptx = _save_review_figure_pair(
            fig, figure_root / f"{FIGURE_STEMS[dataset]}.png"
        )
    finally:
        plt.close(fig)
    if png is None or pptx is None:
        raise RuntimeError(
            f"Figure stem is not registered for canonical export: "
            f"{FIGURE_STEMS[dataset]}"
        )
    return Path(png), Path(pptx)


def _input_stat_audits() -> dict[str, object]:
    audits = {}
    for dataset in ("GSE182434", "Roider"):
        path = Path(fsv.DATASETS[dataset]["h5ad"])
        stat = path.stat()
        audits[dataset] = {
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns_before": int(stat.st_mtime_ns),
            "mtime_ns_after": int(stat.st_mtime_ns),
            "input_unchanged": True,
            "umap_read": False,
        }
    return audits


def _finalize_partial_run(
    run_id: str,
    *,
    source_validation: dict[str, object],
    input_audits: dict[str, object],
    arial_regular: Path,
    arial_bold: Path,
) -> dict[str, object]:
    table_stage = (SENSITIVITY_ROOT / run_id).with_name(run_id + ".partial")
    figure_stage = (FIGURE_ROOT / run_id).with_name(run_id + ".partial")
    table_final = SENSITIVITY_ROOT / run_id
    figure_final = FIGURE_ROOT / run_id
    if table_final.exists() or figure_final.exists():
        raise FileExistsError(f"Completed sensitivity output already exists: {run_id}")
    tables = table_stage / "tables"
    required = {
        "three_arm_pooled_cell_metrics.csv",
        "three_arm_sample_patient_metrics.csv",
        "alias_direction_conflict_audit.csv",
        "three_arm_pairwise_metric_deltas.csv",
    }
    missing = sorted(name for name in required if not (tables / name).is_file())
    if missing:
        raise FileNotFoundError(f"Incomplete sensitivity partial tables: {missing}")
    figure_stage.mkdir(parents=True, exist_ok=True)
    pooled_metrics = pd.read_csv(tables / "three_arm_pooled_cell_metrics.csv")
    grouped_metrics = pd.read_csv(tables / "three_arm_sample_patient_metrics.csv")
    alias_conflicts = pd.read_csv(tables / "alias_direction_conflict_audit.csv")
    expected_rows = len(CONTRACTS) * 2 * 2 * len(METHODS)
    if len(pooled_metrics) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} pooled metric rows, found {len(pooled_metrics)}"
        )
    observed_keys = set(
        map(
            tuple,
            pooled_metrics[
                ["ablation_contract", "dataset", "task", "method"]
            ].astype(str).to_numpy(),
        )
    )
    expected_keys = {
        (contract, dataset, task, method)
        for contract in CONTRACTS
        for dataset in ("GSE182434", "Roider")
        for task in ("broad", "tcell_subtype")
        for method in METHODS
    }
    if observed_keys != expected_keys:
        raise RuntimeError("Sensitivity pooled metric key set is incomplete")
    if len(alias_conflicts) != 4:
        raise RuntimeError(
            f"Expected four dataset-specific alias-conflict rows, found {len(alias_conflicts)}"
        )
    resolved_font = fsv._configure_font(Path(arial_regular), Path(arial_bold))
    figure_paths = [
        figure_stage / f"{FIGURE_STEMS[dataset]}{suffix}"
        for dataset in ("GSE182434", "Roider")
        for suffix in (".png", ".pptx")
    ]
    existing_figure_paths = [path for path in figure_paths if path.is_file()]
    if existing_figure_paths and len(existing_figure_paths) != len(figure_paths):
        raise RuntimeError(
            "Partial Step11B Figure set is incomplete; preserve it for diagnosis and "
            "resume only after resolving the failed pair"
        )
    if existing_figure_paths:
        from Final_utils import _validate_staged_final_png, _validate_staged_final_pptx

        for path in figure_paths:
            if path.suffix.lower() == ".png":
                _validate_staged_final_png(path)
            else:
                _validate_staged_final_pptx(path)
    else:
        figure_paths = []
        for dataset in ("GSE182434", "Roider"):
            figure_paths.extend(
                _save_performance_figure(dataset, pooled_metrics, figure_stage)
            )
    artifacts = []
    for root, kind in ((table_stage, "table_bundle"), (figure_stage, "figure_bundle")):
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.name == "run_manifest.json":
                continue
            artifacts.append(
                {
                    "kind": kind,
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    source_files = [
        SOURCE_TABLES / "combined_fixed_class_direction_signature_features.csv",
        SOURCE_TABLES / "GSE182434_signature_proxy_weights.csv",
        SOURCE_TABLES / "Roider_signature_proxy_weights.csv",
        SOURCE_TABLES / "pooled_cell_metrics.csv",
    ]
    manifest = {
        "status": "complete",
        "analysis_family": "fixed_signature_three_arm_sensitivity",
        "run_id": run_id,
        "source_run_id": fsv.CANONICAL_RUN_ID,
        "canonical_bundle_modified": False,
        "umap_read": False,
        "umap_modified": False,
        "scrna_fitting": False,
        "model_training": False,
        "ale_recalculation": False,
        "permutation_importance_recalculation": False,
        "contracts": list(CONTRACTS),
        "methods": {
            "marker_only": "equal positive weight on fixed ML-derived covered proxy support",
            "collapsed_sign_only": (
                "equal absolute magnitude using the post-alias-collapse ALE sign; "
                "two mixed-direction PTPRC signs inherit importance-weighted alias resolution"
            ),
            "sign_plus_importance": (
                "post-collapse ALE sign plus saved class-normalized permutation-importance "
                "magnitude with signed L1 normalization"
            ),
        },
        "comparison_interpretation": {
            "marker_to_sign": (
                "association with adding post-collapse signed direction under the existing "
                "alias-resolution contract; not a fully importance-free direction effect"
            ),
            "sign_to_importance": (
                "incremental proxy-magnitude reweighting with collapsed signs fixed"
            ),
        },
        "metric_scope": "pooled evaluable cells; descriptive, no patient-level inference",
        "outside_predictions_dropped": False,
        "dataset_wide_proxy_zscore": "label-free transductive preprocessing",
        "resolved_font_family": resolved_font,
        "arial_regular": str(Path(arial_regular).resolve()),
        "arial_bold": str(Path(arial_bold).resolve()),
        "source_validation": json.loads(json.dumps(source_validation, default=str)),
        "input_audits": input_audits,
        "source_hashes": {
            str(path.relative_to(FINAL_ROOT)): _sha256(path) for path in source_files
        },
        "output_roots": {
            "tables": str(table_final),
            "figures": str(figure_final),
        },
        "artifacts": artifacts,
    }
    fsv._write_exact_derived_json(
        table_stage / "run_manifest.json",
        manifest,
        owner="Step11B three-arm sensitivity manifest",
    )
    os.replace(table_stage, table_final)
    try:
        os.replace(figure_stage, figure_final)
    except BaseException:
        os.replace(table_final, table_stage)
        raise
    final_figure_paths = [
        str(figure_final / Path(path).relative_to(figure_stage)) for path in figure_paths
    ]
    return {
        "status": "complete",
        "run_id": run_id,
        "tables": str(table_final),
        "figures": str(figure_final),
        "pooled_metrics_rows": int(len(pooled_metrics)),
        "group_metrics_rows": int(len(grouped_metrics)),
        "alias_conflict_rows": int(len(alias_conflicts)),
        "figure_files": final_figure_paths,
    }


def resume_partial_run(
    run_id: str,
    *,
    arial_regular: Path = DEFAULT_ARIAL_REGULAR,
    arial_bold: Path = DEFAULT_ARIAL_BOLD,
) -> dict[str, object]:
    if fsv.RUN_NAME.fullmatch(run_id or "") is None:
        raise ValueError(f"Unsafe sensitivity run ID: {run_id!r}")
    table_stage = (SENSITIVITY_ROOT / run_id).with_name(run_id + ".partial")
    figure_stage = (FIGURE_ROOT / run_id).with_name(run_id + ".partial")
    if not table_stage.is_dir() or not figure_stage.is_dir():
        raise FileNotFoundError(f"Sensitivity partial bundle is missing: {run_id}")
    source_validation = fsv.validate_completed_run(verify_input_hashes=True)
    return _finalize_partial_run(
        run_id,
        source_validation=source_validation,
        input_audits=_input_stat_audits(),
        arial_regular=arial_regular,
        arial_bold=arial_bold,
    )


def run_sensitivity(
    run_id: str,
    *,
    arial_regular: Path = DEFAULT_ARIAL_REGULAR,
    arial_bold: Path = DEFAULT_ARIAL_BOLD,
) -> dict[str, object]:
    _require_new_run_id(run_id)
    source_validation = fsv.validate_completed_run(verify_input_hashes=True)
    component_directions = _load_component_directions()
    table_stage = (SENSITIVITY_ROOT / run_id).with_name(run_id + ".partial")
    figure_stage = (FIGURE_ROOT / run_id).with_name(run_id + ".partial")
    table_final = SENSITIVITY_ROOT / run_id
    figure_final = FIGURE_ROOT / run_id
    tables = table_stage / "tables"
    tables.mkdir(parents=True, exist_ok=False)
    figure_stage.mkdir(parents=True, exist_ok=False)
    pooled_parts: list[pd.DataFrame] = []
    grouped_parts: list[pd.DataFrame] = []
    alias_audits: list[pd.DataFrame] = []
    input_audits: dict[str, object] = {}
    for dataset in ("GSE182434", "Roider"):
        obs, input_audit = _load_obs_without_umap(dataset)
        input_audits[dataset] = input_audit
        for contract in CONTRACTS:
            weights, alias_audit = _prepare_proxy_weights(
                dataset, component_directions, contract=contract
            )
            if contract == "canonical_compatible":
                alias_audits.append(alias_audit)
            predictions = _score_three_arms(obs, weights)
            pooled, grouped, confusions, reports = _evaluate_three_arms(
                dataset, obs, predictions, contract=contract
            )
            pooled_parts.append(pooled)
            grouped_parts.append(grouped)
            prefix = f"{dataset}_{contract}"
            weights.to_csv(tables / f"{prefix}_proxy_weights.csv", index=False)
            predictions.to_csv(
                tables / f"{prefix}_cell_predictions.csv.gz",
                index=False,
                compression="gzip",
            )
            for key, frame in confusions.items():
                frame.to_csv(tables / f"{contract}_{key}_confusion_matrix.csv")
                row_sums = frame.sum(axis=1).replace(0, np.nan)
                frame.div(row_sums, axis=0).fillna(0.0).to_csv(
                    tables / f"{contract}_{key}_confusion_matrix_row_normalized.csv"
                )
            for key, frame in reports.items():
                frame.to_csv(tables / f"{contract}_{key}_classification_report.csv")
    pooled_metrics = pd.concat(pooled_parts, ignore_index=True)
    grouped_metrics = pd.concat(grouped_parts, ignore_index=True)
    alias_conflicts = pd.concat(alias_audits, ignore_index=True)
    deltas = _pairwise_deltas(pooled_metrics)
    pooled_metrics.to_csv(tables / "three_arm_pooled_cell_metrics.csv", index=False)
    grouped_metrics.to_csv(tables / "three_arm_sample_patient_metrics.csv", index=False)
    alias_conflicts.to_csv(tables / "alias_direction_conflict_audit.csv", index=False)
    deltas.to_csv(tables / "three_arm_pairwise_metric_deltas.csv", index=False)
    return _finalize_partial_run(
        run_id,
        source_validation=source_validation,
        input_audits=input_audits,
        arial_regular=arial_regular,
        arial_bold=arial_bold,
    )


# %% [markdown]
# ## 11B-1. Run configuration
#
# Notebook execution is fail-closed by default. Set `RUN_SENSITIVITY=True` and
# choose a new run ID to create another sensitivity bundle.

# %%
RUN_SENSITIVITY = False
SENSITIVITY_RUN_ID = DEFAULT_RUN_ID


# %% [markdown]
# ## 11B-2. Execute and summarize

# %%
if RUN_SENSITIVITY and "ipykernel" in sys.modules:
    notebook_report = run_sensitivity(SENSITIVITY_RUN_ID)
    print(json.dumps(notebook_report, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--arial-regular", type=Path, default=DEFAULT_ARIAL_REGULAR)
    parser.add_argument("--arial-bold", type=Path, default=DEFAULT_ARIAL_BOLD)
    parser.add_argument("--resume-partial", action="store_true")
    args = parser.parse_args(argv)
    runner = resume_partial_run if args.resume_partial else run_sensitivity
    report = runner(
        args.run_id,
        arial_regular=args.arial_regular,
        arial_bold=args.arial_bold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
