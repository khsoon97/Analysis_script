# Reproduction scope and computational references

`reproduce.py` reads only the pinned CSV for the selected supported panel. It reshapes saved tables, plots saved values and translates a few display labels; it does not reselect cells, refit models, normalize scores, recalculate uncertainty or run tests. The paired-distance, nearest-only, pooled-cell and fixed-signature definitions remain in the reference methods. In Figure S6e the source is the current mean-difference table, not a historical median-difference table. Source coverage does not establish patient-level inference or independent phenotype validation.

## Reference implementations

| Sources in `reference_source/` | Purpose / relevant Figures |
|---|---|
| `Final_step00_figure1A_graphical_abstract.py`, `Final_step01.py` | Study schematic, CODEX composition; Figure 1 (author edits not reproduced) |
| `Final_step02.py` | Broad-lineage training/rescue; Figures 2, S1, S2 |
| `Final_step03.py` | Broad-lineage benchmark; Figures 3, S2 |
| `Final_step04B.py` | Lineage-routed subtype classification and attribution; Figures 4, S3, S4 |
| `Final_step05.py` | Inactive historical CN/NMF upstream source, not a Figure replay step |
| `Final_step06.py` | Saved spatial distances and functional profiles; Figures 5, S6 |
| `Final_step07.py` | T-cell-anchored motifs; Figures 6, S6 |
| `Final_step08.py` | Saved marker-product interaction enrichment and correlations; Figures 5, S6 |
| `Final_step09*`, `Final_step10*` | External scRNA processing source reference |
| `Final_step11*`, `Final_step11B*`, `Final_step12*` | Fixed CODEX-derived signature validation, sign sensitivity and export; Figure S5 |
| Remaining modules | Direct dependencies, including shared Figure export and model-attribution implementations |

Source hashes, portable-string conversion provenance, output-helper relocations and English-language translation are recorded in `manifests/sources.json`. The analysis implementations retain their separate scientific inputs, masks and model semantics. Only shared output infrastructure is consolidated in `editable_pptx.py`: `validate_final_figure_canvas`, `_validate_staged_final_png`, `_validate_staged_final_pptx`, `_new_final_pair_transaction_paths`, `_promote_staged_final_pair` and `_configure_font`. The original modules directly import these names, preserving their callers. Analysis, training and evaluation functions are not merged. Output-path authorization and the distinct legacy A4 export behavior remain with their original callers.

The computational sources are not imported by the renderer. They contain top-level analysis code and path placeholders (`__ROIDER_ROOT__`, `__FONT_ROOT__`, `__PYTHON__`, `__TEMP_ROOT__`); running them directly is unsupported. A computational rerun requires an independently prepared workspace, audited output routes, the selected upstream datasets and pinned model/checkpoint/signature inputs. No training or original computational rerun was performed for this Figure-replay release. Relocating output helpers does not revalidate or rewrite the provenance of previously completed analysis runs.

`manifests/reference_inputs.json` records known computational input boundaries, not a complete leaf-level manifest. Public source identifiers are BioImage Archive S-BIAD565 for CODEX and VRJUNV/GSE182434 for scRNA. Those public accessions do not themselves supply every processed checkpoint, selected probability cache or attribution table used by this manuscript. Fixed-signature RNA validation does not transfer fitted spatial classifier objects or run spatial rescue.

## Environment boundaries

`requirements.txt` is the small saved-table plotting environment (NumPy, pandas, Matplotlib). No GPU packages are needed for that entry point. `manifests/reference_environment.json` reports previously observed distribution versions only for packages directly imported by the retained CODEX/scRNA sources. It is **not** a clean-machine solved lockfile or proof of the historical environment. Full model code has additional CPU/GPU and vendor CUDA constraints. Original native editable PPTX export in `editable_pptx.py` also has Microsoft PowerPoint/platform dependencies; portable replay instead writes SVG and PNG.

## Current manuscript authority

The shared manuscript Figure masters are maintained separately by the authors. They contain subsequent panel label, HLA class II terminology, radius annotation, mean-display and manual-composite edits. The reference Final source snapshot is methodological provenance, not a claim that every historical display string matches the current edited master. The replay has explicit current-table routing and its own output label. It does not overwrite or automatically rebuild the shared masters.

## English-language publication copy

Korean comments, docstrings and human-readable log/error messages were translated to English. Static comparison preserves executable structure, f-string expressions and numeric settings except for translated display strings. The legacy layout-reference filename retains its exact Unicode value through escaped source text. These language changes do not alter analytical definitions, data keys or input/output paths, and no analysis was rerun for this publication copy.
