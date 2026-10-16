# Canonical script; synchronized to the paired notebook via scripts/sync_notebooks.py.
# %% [markdown]
# # Final Step 11. Frozen fixed-signature scRNA validation
#
# Primary Final contract: marker-only versus marker + DLBCL-ALE direction/weight. This is neither a frozen classifier-object transfer nor V8 scRNA-supervised OOF retraining.

# %% [markdown]
# ## 11-0. Execution contract
#
# Default execution reuses protected proxy inputs and pinned Step02/04B source
# results, then replaces only the canonical derived signature, annotation,
# Z-score, and metric bundle. Raw preprocessing and proxy generation remain a
# separate opt-in path shared with Steps09/10.

# %%
from pathlib import Path
import os
import pandas as pd
from IPython.display import display

from final_fixed_signature_validation import CANONICAL_RUN_ID
from final_fixed_signature_validation import COMPLETED_BUNDLE_ROOT as CANONICAL_ROOT
from final_fixed_signature_validation import main as fixed_signature_main
from final_fixed_signature_validation import require_new_run_id
from final_fixed_signature_validation import validate_completed_run

FINAL_ROOT = Path('__ROIDER_ROOT__/Analysis/python/Final')
os.environ['FINAL_FIGURE_ONLY_WRITE_GUARD'] = '1'
os.environ['SCRNA_FIGURE_ONLY_WRITE_GUARD'] = '1'
os.environ['SCRNA_WRITE_AUDIT_TABLES'] = '0'
SCRNA_RECALCULATE = False
SCRNA_NEW_RUN_ID = None
SCRNA_RECOMPUTE_CANONICAL_DERIVED = False

# %% [markdown]
# ## 11-1. Active run and denominator contract
#
# Both methods share a truth-derived mask. Broad truth is `reference_broad_codex`; subtype truth is gated by reference broad T cells and restricted to TH/TFH/TREG/CD8T. TPR, CD4_CTL, and Unassigned predictions remain denominator errors.

# %%
if SCRNA_RECALCULATE:
    require_new_run_id(SCRNA_NEW_RUN_ID)
    handoff_root = (
        FINAL_ROOT / 'tables' / 'scRNA' / 'fixed_signature_recalculation_inputs'
        / SCRNA_NEW_RUN_ID
    )
    fixed_signature_main([
        '--force-recalculate',
        '--run-id', SCRNA_NEW_RUN_ID,
        '--roider-input-manifest', str(handoff_root / 'Roider' / 'step09_input_manifest.json'),
        '--gse182434-input-manifest', str(handoff_root / 'GSE182434' / 'step10_input_manifest.json'),
    ])
    active_run_id = SCRNA_NEW_RUN_ID
    active_root = FINAL_ROOT / 'tables' / 'scRNA' / 'fixed_signature' / active_run_id
elif SCRNA_RECOMPUTE_CANONICAL_DERIVED:
    fixed_signature_main([
        '--replace-canonical-derived',
        '--run-id', CANONICAL_RUN_ID,
    ])
    active_run_id = CANONICAL_RUN_ID
    active_root = CANONICAL_ROOT
    validate_completed_run(verify_input_hashes=True)
else:
    active_run_id = CANONICAL_RUN_ID
    active_root = CANONICAL_ROOT
    validate_completed_run(verify_input_hashes=True)

pooled_metrics = pd.read_csv(active_root / 'tables' / 'pooled_cell_metrics.csv')

# %% [markdown]
# ## 11-2. Completed Roider predictions and masks

# %%
roider_predictions = pd.read_csv(
    active_root / 'tables' / 'Roider_cell_predictions.csv.gz', dtype={'cell_id': str}
)
roider_mask_audit = pd.read_csv(
    active_root / 'tables' / 'Roider_evaluation_mask_audit.csv.gz', dtype={'cell_id': str}
)
roider_metrics = pooled_metrics.loc[pooled_metrics['dataset'].eq('Roider')].copy()
print('Roider completed cells:', len(roider_predictions))
display(roider_metrics)

# %% [markdown]
# ## 11-3. Completed GSE182434 predictions and masks

# %%
gse182434_predictions = pd.read_csv(
    active_root / 'tables' / 'GSE182434_cell_predictions.csv.gz', dtype={'cell_id': str}
)
gse182434_mask_audit = pd.read_csv(
    active_root / 'tables' / 'GSE182434_evaluation_mask_audit.csv.gz', dtype={'cell_id': str}
)
gse182434_metrics = pooled_metrics.loc[pooled_metrics['dataset'].eq('GSE182434')].copy()
print('GSE182434 completed cells:', len(gse182434_predictions))
display(gse182434_metrics)

# %% [markdown]
# ## 11-4. Execution summary
#
# A new run is accepted only through the paired Step09/10 manifests. The backend validates the completed immutable run before this summary is displayed.

# %%
if SCRNA_RECALCULATE:
    print(f'Step11 new run validated: {active_run_id}')
elif SCRNA_RECOMPUTE_CANONICAL_DERIVED:
    print(
        'Step11 canonical derived outputs replaced: '
        'raw preprocessing=False; proxy recompute=False; model training=False; ALE=False'
    )
else:
    print('Step11 reuse-only: completed predictions/metrics reused; scoring=False; metrics=False')
validation_summary = pd.DataFrame([
    {
        'dataset': dataset,
        'run_id': active_run_id,
        'analysis_family': 'frozen_fixed_signature',
        'methods': 'marker_only|marker_ale_direction_weighted',
        'truth_derived_shared_mask': True,
        'outside_predictions_dropped': False,
        'classifier_object_transfer': False,
        'scrna_supervised_retraining': False,
    }
    for dataset in ('Roider', 'GSE182434')
])
display(validation_summary)
