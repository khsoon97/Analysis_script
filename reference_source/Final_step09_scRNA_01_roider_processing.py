# Canonical script; synchronized to the paired notebook via scripts/sync_notebooks.py.
# %% [markdown]
# # Final Step 09. Roider scRNA fixed-signature input preparation
#
# By default, read only canonical Roider results copied to and verified in Final and their input contracts. Raw-data QC and gene-proxy calculations run only in an explicit new run.

# %% [markdown]
# ## 09-0. Execution contract
#
# `SCRNA_RECALCULATE=False` is the default. Recalculation requires both `True` and a unique `SCRNA_NEW_RUN_ID` that differs from the canonical ID.

# %%
from pathlib import Path
import json
import os

FINAL_ROOT = Path('__ROIDER_ROOT__/Analysis/python/Final')
os.environ['FINAL_FIGURE_ONLY_WRITE_GUARD'] = '1'
os.environ['SCRNA_FIGURE_ONLY_WRITE_GUARD'] = '1'
os.environ['SCRNA_WRITE_AUDIT_TABLES'] = '0'
from final_fixed_signature_validation import CANONICAL_RUN_ID
from final_fixed_signature_validation import COMPLETED_BUNDLE_ROOT as CANONICAL_ROOT
from final_fixed_signature_validation import require_new_run_id
from final_fixed_signature_validation import validate_completed_run
from final_fixed_signature_validation import write_prepared_input_manifest

SCRNA_RECALCULATE = False
SCRNA_NEW_RUN_ID = None

# %% [markdown]
# ## 09-1. Check completed canonical Roider results
#
# This step does not regenerate predictions. Check the Final-local prediction, mask, coverage, and metric contracts consumed by Step11.

# %%
canonical_report = validate_completed_run(
    verify_input_hashes=True,
    canonical_proxy_datasets=('Roider',),
)
roider_required = {
    'predictions': CANONICAL_ROOT / 'tables' / 'Roider_cell_predictions.csv.gz',
    'scores': CANONICAL_ROOT / 'tables' / 'Roider_class_scores.csv.gz',
    'mask_audit': CANONICAL_ROOT / 'tables' / 'Roider_evaluation_mask_audit.csv.gz',
    'coverage_audit': CANONICAL_ROOT / 'tables' / 'Roider_signature_proxy_coverage_audit.csv',
    'proxy_weights': CANONICAL_ROOT / 'tables' / 'Roider_signature_proxy_weights.csv',
}
missing = [str(path) for path in roider_required.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(f'Canonical Roider fixed-signature files missing: {missing}')
print('Step09 default: canonical Final-local Roider results reused; preprocessing=False')

# %% [markdown]
# ## 09-2. Explicit future recalculation path
#
# This cell uses only preprocessing/proxy code owned by Final. Stop without overwriting if the checkpoint/table folders for the unique run already exist.

# %%
if SCRNA_RECALCULATE:
    require_new_run_id(SCRNA_NEW_RUN_ID)

    input_root = FINAL_ROOT / 'checkpoints' / 'scRNA' / 'recalculation_inputs' / SCRNA_NEW_RUN_ID / 'Roider'
    table_root = FINAL_ROOT / 'tables' / 'scRNA' / 'fixed_signature_recalculation_inputs' / SCRNA_NEW_RUN_ID / 'Roider'
    if input_root.exists() or table_root.exists():
        raise FileExistsError(f'Refusing to overwrite existing recalculation inputs: {SCRNA_NEW_RUN_ID}')

    os.environ['SCRNA_FIGURE_ONLY_WRITE_GUARD'] = '0'
    os.environ['SCRNA_WRITE_AUDIT_TABLES'] = '1'
    import direct_transfer_validation_utils as dtv

    dtv.SCRNA_FIGURE_ONLY_WRITE_GUARD = False
    dtv.SCRNA_WRITE_AUDIT_TABLES = True
    dtv.OUT_DIR = table_root
    dtv.ROIDER_CACHE_DIR = input_root
    dtv.ROIDER_CURATED_ANNOTATION_DIR = table_root / 'curated_annotations'
    dtv.B_ANNOTATION_CURATED = dtv.ROIDER_CURATED_ANNOTATION_DIR / 'AnnotationBcells_curated.csv'
    dtv.T_ANNOTATION_CURATED = dtv.ROIDER_CURATED_ANNOTATION_DIR / 'Annotation_Tcells_curated.csv'
    dtv.B_ANNOTATION = dtv.B_ANNOTATION_CURATED
    dtv.T_ANNOTATION = dtv.T_ANNOTATION_CURATED
    dtv.RAW_H5AD = input_root / '01_roider_raw_counts_with_reference.h5ad'
    dtv.PROCESSED_H5AD = input_root / '01_roider_processed_umap.h5ad'
    dtv.FEATURE_H5AD = input_root / '01_roider_codex_proxy_features.h5ad'
    routed_roots = {
        'OUT_DIR': (Path(dtv.OUT_DIR).resolve(), table_root.resolve()),
        'ROIDER_CACHE_DIR': (Path(dtv.ROIDER_CACHE_DIR).resolve(), input_root.resolve()),
        'RAW_H5AD': (Path(dtv.RAW_H5AD).resolve().parent, input_root.resolve()),
        'PROCESSED_H5AD': (Path(dtv.PROCESSED_H5AD).resolve().parent, input_root.resolve()),
        'FEATURE_H5AD': (Path(dtv.FEATURE_H5AD).resolve().parent, input_root.resolve()),
    }
    mismatched_routes = [name for name, (actual, expected) in routed_roots.items() if actual != expected]
    if mismatched_routes:
        raise RuntimeError(f'Step09 isolated output routing failed: {mismatched_routes}')

    signature_table = CANONICAL_ROOT / 'tables' / 'combined_fixed_class_direction_signature_features.csv'
    signature = __import__('pandas').read_csv(signature_table)
    feature_names = (
        signature.sort_values(['feature_position', 'feature'], kind='mergesort')['feature']
        .drop_duplicates().astype(str).tolist()
    )
    adata_feature, coverage, sample_sheet, reference_summary = dtv.prepare_roider_feature_from_raw(
        feature_names,
        force_reload_raw=True,
        force_reprocess=True,
        force_recompute_features=True,
        feature_h5ad=dtv.FEATURE_H5AD,
        out_dir=dtv.OUT_DIR,
    )
    step09_manifest_path = table_root / 'step09_input_manifest.json'
    step09_manifest = write_prepared_input_manifest(
        step09_manifest_path,
        stage='Step09',
        run_id=SCRNA_NEW_RUN_ID,
        dataset='Roider',
        feature_h5ad=dtv.FEATURE_H5AD,
        coverage_csv=table_root / 'tables' / 'codex_gene_proxy_coverage.csv',
        signature_table=signature_table,
        feature_names=feature_names,
        adata=adata_feature,
    )
else:
    step09_manifest = {
        'status': 'reused_canonical',
        'run_id': CANONICAL_RUN_ID,
        'preprocessing_executed': False,
        'proxy_features_recomputed': False,
    }
    step09_manifest_path = None

# %% [markdown]
# ## 09-3. Step11 handoff
#
# The default handoff uses Final-local canonical tables. Pass unique feature/coverage paths to Step11 only for a new calculation.

# %%
step09_handoff = {
    'dataset': 'Roider',
    'mode': 'recalculate' if SCRNA_RECALCULATE else 'reuse_canonical',
    'run_id': SCRNA_NEW_RUN_ID if SCRNA_RECALCULATE else CANONICAL_RUN_ID,
    'input_manifest': str(step09_manifest_path) if step09_manifest_path else None,
    'canonical_prediction_table': str(roider_required['predictions']),
    'canonical_evaluation_mask': str(roider_required['mask_audit']),
}
print(json.dumps(step09_handoff, ensure_ascii=False, indent=2))
