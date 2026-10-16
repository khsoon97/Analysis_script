# Canonical script; synchronized to the paired notebook via scripts/sync_notebooks.py.
# %% [markdown]
# # Final Step 10. GSE182434 scRNA fixed-signature input preparation
#
# By default, read only canonical GSE182434 results copied to and verified in Final and their input contracts. Raw-data QC and gene-proxy calculations run only in an explicit new run.

# %% [markdown]
# ## 10-0. Execution contract
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
# ## 10-1. Check completed canonical GSE182434 results
#
# This step does not regenerate predictions. Check the Final-local prediction, mask, coverage, and metric contracts consumed by Step11.

# %%
canonical_report = validate_completed_run(
    verify_input_hashes=True,
    canonical_proxy_datasets=('GSE182434',),
)
gse_required = {
    'predictions': CANONICAL_ROOT / 'tables' / 'GSE182434_cell_predictions.csv.gz',
    'scores': CANONICAL_ROOT / 'tables' / 'GSE182434_class_scores.csv.gz',
    'mask_audit': CANONICAL_ROOT / 'tables' / 'GSE182434_evaluation_mask_audit.csv.gz',
    'coverage_audit': CANONICAL_ROOT / 'tables' / 'GSE182434_signature_proxy_coverage_audit.csv',
    'proxy_weights': CANONICAL_ROOT / 'tables' / 'GSE182434_signature_proxy_weights.csv',
}
missing = [str(path) for path in gse_required.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(f'Canonical GSE182434 fixed-signature files missing: {missing}')
print('Step10 default: canonical Final-local GSE182434 results reused; preprocessing=False')

# %% [markdown]
# ## 10-2. Explicit future recalculation path
#
# This cell uses only preprocessing/proxy code owned by Final. Stop without overwriting if the checkpoint/table folders for the unique run already exist.

# %%
if SCRNA_RECALCULATE:
    require_new_run_id(SCRNA_NEW_RUN_ID)

    input_root = FINAL_ROOT / 'checkpoints' / 'scRNA' / 'recalculation_inputs' / SCRNA_NEW_RUN_ID / 'GSE182434'
    table_root = FINAL_ROOT / 'tables' / 'scRNA' / 'fixed_signature_recalculation_inputs' / SCRNA_NEW_RUN_ID / 'GSE182434'
    if input_root.exists() or table_root.exists():
        raise FileExistsError(f'Refusing to overwrite existing recalculation inputs: {SCRNA_NEW_RUN_ID}')

    os.environ['SCRNA_FIGURE_ONLY_WRITE_GUARD'] = '0'
    os.environ['SCRNA_WRITE_AUDIT_TABLES'] = '1'
    import direct_transfer_validation_utils as dtv

    dtv.SCRNA_FIGURE_ONLY_WRITE_GUARD = False
    dtv.SCRNA_WRITE_AUDIT_TABLES = True
    dtv.GSE182434_OUT_DIR = table_root
    dtv.GSE182434_CACHE_DIR = input_root
    dtv.GSE182434_QC_H5AD = input_root / '00_gse182434_qc_processed_umap.h5ad'
    dtv.GSE182434_FEATURE_H5AD = input_root / '01_gse182434_codex_proxy_features.h5ad'
    routed_roots = {
        'GSE182434_OUT_DIR': (Path(dtv.GSE182434_OUT_DIR).resolve(), table_root.resolve()),
        'GSE182434_CACHE_DIR': (Path(dtv.GSE182434_CACHE_DIR).resolve(), input_root.resolve()),
        'GSE182434_QC_H5AD': (Path(dtv.GSE182434_QC_H5AD).resolve().parent, input_root.resolve()),
        'GSE182434_FEATURE_H5AD': (
            Path(dtv.GSE182434_FEATURE_H5AD).resolve().parent,
            input_root.resolve(),
        ),
    }
    mismatched_routes = [name for name, (actual, expected) in routed_roots.items() if actual != expected]
    if mismatched_routes:
        raise RuntimeError(f'Step10 isolated output routing failed: {mismatched_routes}')

    signature_table = CANONICAL_ROOT / 'tables' / 'combined_fixed_class_direction_signature_features.csv'
    signature = __import__('pandas').read_csv(signature_table)
    feature_names = (
        signature.sort_values(['feature_position', 'feature'], kind='mergesort')['feature']
        .drop_duplicates().astype(str).tolist()
    )
    adata_gse, reference_summary = dtv.preprocess_gse182434_for_transfer(force_reprocess=True)
    adata_feature, coverage = dtv.add_gse182434_codex_proxy_features(
        feature_names, force_recompute=True, a=adata_gse
    )
    step10_manifest_path = table_root / 'step10_input_manifest.json'
    step10_manifest = write_prepared_input_manifest(
        step10_manifest_path,
        stage='Step10',
        run_id=SCRNA_NEW_RUN_ID,
        dataset='GSE182434',
        feature_h5ad=dtv.GSE182434_FEATURE_H5AD,
        coverage_csv=table_root / 'tables' / 'codex_gene_proxy_coverage.csv',
        signature_table=signature_table,
        feature_names=feature_names,
        adata=adata_feature,
    )
else:
    step10_manifest = {
        'status': 'reused_canonical',
        'run_id': CANONICAL_RUN_ID,
        'preprocessing_executed': False,
        'proxy_features_recomputed': False,
    }
    step10_manifest_path = None

# %% [markdown]
# ## 10-3. Step11 handoff
#
# The default handoff uses Final-local canonical tables. Pass unique feature/coverage paths to Step11 only for a new calculation.

# %%
step10_handoff = {
    'dataset': 'GSE182434',
    'mode': 'recalculate' if SCRNA_RECALCULATE else 'reuse_canonical',
    'run_id': SCRNA_NEW_RUN_ID if SCRNA_RECALCULATE else CANONICAL_RUN_ID,
    'input_manifest': str(step10_manifest_path) if step10_manifest_path else None,
    'canonical_prediction_table': str(gse_required['predictions']),
    'canonical_evaluation_mask': str(gse_required['mask_audit']),
}
print(json.dumps(step10_handoff, ensure_ascii=False, indent=2))
