# Canonical script; synchronized to the paired notebook via scripts/sync_notebooks.py.
# %% [markdown]
# # Final Step 12. Fixed-signature Figure export/reuse
#
# Default execution renders from the current canonical derived prediction/metric
# bundle plus the pinned Step11B three-arm sensitivity bundle and replaces the
# complete active Final PNG/PPTX set. Scoring, metrics, raw preprocessing,
# model training, permutation importance, and ALE are never run here.

# %% [markdown]
# ## 12-0. Execution contract
#
# Figure-only execution renders the complete active registered set. By default
# the script validates and reuses the current canonical files without writing.
# Set `SCRNA_RENDER_COMPLETED_FIGURES=1` plus a new `SCRNA_FIGURE_EXPORT_TAG` for
# an isolated review export. Canonical replacement additionally requires
# `SCRNA_PROMOTE_CANONICAL_FIGURES=1` and always promotes all four pairs together.

# %%
from pathlib import Path
import json
import os
import pandas as pd
from IPython.display import display

from final_fixed_signature_validation import CANONICAL_RUN_ID
from final_fixed_signature_validation import CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID
from final_fixed_signature_validation import FINAL_SCRNA_A4_COMPOSITE_STEM
from final_fixed_signature_validation import render_figures_from_completed_run
from final_fixed_signature_validation import render_a4_composite_from_completed_run
from final_fixed_signature_validation import RUN_NAME
from final_fixed_signature_validation import validate_completed_run
from final_fixed_signature_validation import FINAL_SCRNA_FIGURE_STEMS


def _scrna_env_flag(name, default=False):
    raw = os.environ.get(name, '1' if default else '0').strip()
    if raw not in {'0', '1'}:
        raise ValueError(f'{name} must be 0 or 1, got {raw!r}')
    return raw == '1'


FINAL_ROOT = Path('__ROIDER_ROOT__/Analysis/python/Final')
os.environ['FINAL_FIGURE_ONLY_WRITE_GUARD'] = '1'
os.environ['SCRNA_FIGURE_ONLY_WRITE_GUARD'] = '1'
os.environ['SCRNA_WRITE_AUDIT_TABLES'] = '0'
os.environ.setdefault('FINAL_LIGHT_IMPORTS', '1')
SCRNA_RENDER_COMPLETED_FIGURES = _scrna_env_flag(
    'SCRNA_RENDER_COMPLETED_FIGURES', False
)
SCRNA_SOURCE_RUN_ID = CANONICAL_RUN_ID
SCRNA_PERFORMANCE_SENSITIVITY_RUN_ID = CANONICAL_THREE_ARM_SENSITIVITY_RUN_ID
SCRNA_FIGURE_EXPORT_TAG = os.environ.get('SCRNA_FIGURE_EXPORT_TAG', '').strip() or None
SCRNA_PROMOTE_CANONICAL_FIGURES = _scrna_env_flag(
    'SCRNA_PROMOTE_CANONICAL_FIGURES', False
)
SCRNA_PROMOTE_A4_2X2_COMPOSITE = _scrna_env_flag(
    'SCRNA_PROMOTE_A4_2X2_COMPOSITE', False
)
if SCRNA_PROMOTE_CANONICAL_FIGURES and not SCRNA_RENDER_COMPLETED_FIGURES:
    raise ValueError(
        'SCRNA_PROMOTE_CANONICAL_FIGURES=1 requires '
        'SCRNA_RENDER_COMPLETED_FIGURES=1'
    )

# %% [markdown]
# ## 12-1. Final-local completed Figure pairs

# %%
panel_specs = (
    ('A', 'Step12-01', 'GSE182434', 'performance'),
    ('B', 'Step12-02', 'GSE182434', '3x3_umap_summary'),
    ('C', 'Step12-03', 'Roider', 'performance'),
    ('D', 'Step12-04', 'Roider', '3x3_umap_summary'),
)
canonical_report = None
completed_figure_index = pd.DataFrame()
if not SCRNA_RENDER_COMPLETED_FIGURES:
    canonical_report = validate_completed_run(verify_input_hashes=True)
    figure_rows = []
    for panel, output_number, dataset, figure_kind in panel_specs:
        dataset_root = FINAL_ROOT / 'figures' / 'v2' / 'scRNA' / dataset
        stem = FINAL_SCRNA_FIGURE_STEMS[(dataset, figure_kind)]
        png = dataset_root / f'{stem}.png'
        pptx = dataset_root / f'{stem}.pptx'
        if not png.is_file() or not pptx.is_file():
            raise FileNotFoundError(f'Completed Final Figure pair missing: {stem}')
        figure_rows.append({
            'panel': panel,
            'output_number': output_number,
            'dataset': dataset,
            'figure_kind': figure_kind,
            'stem': stem,
            'png': str(png),
            'pptx': str(pptx),
            'reused_without_regeneration': True,
            'regenerated_this_run': False,
        })
    completed_figure_index = pd.DataFrame(figure_rows)
    display(completed_figure_index)

# %% [markdown]
# ## 12-2. Explicit Figure-only re-render
#
# The output tag must be new. The renderer loads only a completed Final-local run's saved prediction and metric tables.

# %%
if SCRNA_RENDER_COMPLETED_FIGURES:
    if RUN_NAME.fullmatch(SCRNA_SOURCE_RUN_ID or '') is None:
        raise ValueError('Figure-only rendering requires a safe source run ID.')
    if (
        not SCRNA_PROMOTE_CANONICAL_FIGURES
        and RUN_NAME.fullmatch(SCRNA_FIGURE_EXPORT_TAG or '') is None
    ):
        raise ValueError('Noncanonical Figure export requires a new export tag.')
    render_report = render_figures_from_completed_run(
        SCRNA_SOURCE_RUN_ID,
        export_tag=(
            None if SCRNA_PROMOTE_CANONICAL_FIGURES
            else SCRNA_FIGURE_EXPORT_TAG
        ),
        datasets=['Roider', 'GSE182434'],
        arial_regular=Path('__FONT_ROOT__/arial.ttf'),
        arial_bold=Path('__FONT_ROOT__/arialbd.ttf'),
        overwrite_canonical=SCRNA_PROMOTE_CANONICAL_FIGURES,
        performance_sensitivity_run_id=SCRNA_PERFORMANCE_SENSITIVITY_RUN_ID,
    )
    rendered_root = Path(render_report['output_directory'])
    updated_paths = {
        str(Path(path).resolve()) for path in render_report.get('generated_figures', [])
    }
    figure_rows = []
    for panel, output_number, dataset, figure_kind in panel_specs:
        stem = FINAL_SCRNA_FIGURE_STEMS[(dataset, figure_kind)]
        dataset_root = (
            rendered_root / dataset
            if SCRNA_PROMOTE_CANONICAL_FIGURES
            else rendered_root
        )
        png = dataset_root / f'{stem}.png'
        pptx = dataset_root / f'{stem}.pptx'
        if not png.is_file() or not pptx.is_file():
            raise FileNotFoundError(f'Rendered Figure pair missing: {stem}')
        regenerated_this_run = (
            str(png.resolve()) in updated_paths
            and str(pptx.resolve()) in updated_paths
        )
        figure_rows.append({
            'panel': panel,
            'output_number': output_number,
            'dataset': dataset,
            'figure_kind': figure_kind,
            'stem': stem,
            'png': str(png),
            'pptx': str(pptx),
            'reused_without_regeneration': not regenerated_this_run,
            'regenerated_this_run': regenerated_this_run,
        })
    completed_figure_index = pd.DataFrame(figure_rows)
else:
    render_report = {
        'status': 'reused_completed_final_figures',
        'source_run_id': CANONICAL_RUN_ID,
        'preprocessing_executed': False,
        'scores_recomputed': False,
        'metrics_recomputed': False,
        'figure_regenerated': False,
    }
print(json.dumps(render_report, ensure_ascii=False, indent=2))

# %% [markdown]
# ## 12-3. Layout reference PPTX: A4 portrait 2×2 composite

# %%
a4_composite_report = {
    'status': 'A4_composite_not_requested',
    'figure_regenerated': False,
}
if SCRNA_PROMOTE_A4_2X2_COMPOSITE:
    a4_composite_report = render_a4_composite_from_completed_run(
        SCRNA_SOURCE_RUN_ID,
        performance_sensitivity_run_id=SCRNA_PERFORMANCE_SENSITIVITY_RUN_ID,
        arial_regular=Path('__FONT_ROOT__/arial.ttf'),
        arial_bold=Path('__FONT_ROOT__/arialbd.ttf'),
    )
a4_root = FINAL_ROOT / 'figures' / 'v2' / 'scRNA'
a4_png = a4_root / f'{FINAL_SCRNA_A4_COMPOSITE_STEM}.png'
a4_pptx = a4_root / f'{FINAL_SCRNA_A4_COMPOSITE_STEM}.pptx'
if a4_png.is_file() and a4_pptx.is_file():
    completed_figure_index = pd.concat(
        [
            completed_figure_index,
            pd.DataFrame([
                {
                    'panel': 'A-D',
                    'output_number': 'Step12-05',
                    'dataset': 'GSE182434 × Roider',
                    'figure_kind': 'A4_2x2_composite',
                    'stem': FINAL_SCRNA_A4_COMPOSITE_STEM,
                    'png': str(a4_png),
                    'pptx': str(a4_pptx),
                    'reused_without_regeneration': not bool(
                        a4_composite_report.get('figure_regenerated', False)
                    ),
                    'regenerated_this_run': bool(
                        a4_composite_report.get('figure_regenerated', False)
                    ),
                }
            ]),
        ],
        ignore_index=True,
    )
print(json.dumps(a4_composite_report, ensure_ascii=False, indent=2))

# %% [markdown]
# ## 12-4. Manuscript Figure mapping

# %%
panel_mapping = completed_figure_index.copy()
display(panel_mapping)

# %% [markdown]
# ## 12-5. Safety summary

# %%
figure_summary = pd.DataFrame([render_report, a4_composite_report])
display(figure_summary)

# %% [markdown]
# ## 12-6. Analysis-family boundary
#
# Performance panels report the pinned marker/sign/PFI compatibility ablation;
# UMAP panels retain the completed two-arm Step11 display. Archived V8 OOF
# retraining and the former fitted-classifier direct-transfer route are not active inputs.

# %%
analysis_family_summary = pd.DataFrame([{
    'active_family': 'frozen_fixed_signature',
    'marker_only': True,
    'collapsed_sign_only_compatibility_ablation': True,
    'marker_ale_direction_importance_weighted': True,
    'performance_source_run_id': SCRNA_PERFORMANCE_SENSITIVITY_RUN_ID,
    'umap_source_run_id': SCRNA_SOURCE_RUN_ID,
    'frozen_classifier_object_transfer': False,
    'scrna_supervised_oof': False,
    'font_contract_for_new_render': 'strict Arial Regular/Bold',
}])
display(analysis_family_summary)
