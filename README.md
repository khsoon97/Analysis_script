# Roider manuscript Figure replay

Start with **one entry point, `reproduce.py`**, to draw supported panels from the authors' saved source tables. It never imports the analysis pipeline, fits a model, computes a new statistical test, or writes to supplied inputs. This repository contains **code and input specifications only**; obtain the reviewed source-data collection separately from the authors. Data publication is a separate decision.

```bash
python -m pip install -r requirements.txt
python reproduce.py list
python reproduce.py check
python reproduce.py render --panel S6e \
  --source-data /path/to/source_data \
  --font-dir /path/to/licensed/Arial \
  --output /path/to/new-output/FigureS6e
```

Use Python 3.10 or later. The font directory must contain real Arial Regular `arial.ttf` and Bold `arialbd.ttf`; fonts are not distributed. Each render requires a **new output directory** and writes PNG, editable-text SVG and a short input-hash record. SVG is a vector Figure asset, not a native PowerPoint composite.

## What can be replayed

| Panel | Saved input | Replayed values |
|---|---|---|
| 1b | `figure1b_tissue_lineage_counts.csv` | Tissue lineage percentages |
| 2a | `figure2a_display_marker_model_bubbles.csv` | Selected-marker importance and capped Welch P display |
| 3a | `mv1_step03_benchmark_results.csv` | Broad-lineage benchmark scores |
| 4a | `figure4a_conditional_dlbcl_subtype_benchmark.csv` | Conditional subtype benchmark scores |
| 5c | `figure5c_functional_distance_bin_means.csv` | Pooled means by distance bin |
| S1c | `figureS1c_changed_cells_confusion_counts.csv` | Changed-cell row percentages |
| S3b | `figureS3b_grzb_high_component_posterior_bins.csv` | GZMB posterior-bin curve and saved cutoff |
| S6e | `figureS6e_pooled_mean_differences.csv` | Current pooled **mean** differences |

Exact relative paths, SHA-256 hashes and columns are in [the panel manifest](manifests/figures.json). Input hashes are checked before rendering; a different table is not silently treated as the reviewed input. These plots preserve the saved values but use a compact replay layout. Benchmark replays show all models present in their saved tables. This is **not** pixel-identical reconstruction of author-edited panels or complete manuscript composites. No uncertainty interval, P value or inference is newly calculated.

The manifest also describes unsupported panels and missing input classes. Per-cell UMAP/scatter, microscopy images, some checkpoint-derived displays and edited composites cannot be recovered from aggregate tables alone. They fail as unsupported rather than starting training or reanalysis. See [scope and source mapping](docs/scope.md).

## Where the other code went

`reference_source/` holds **one copy of 25 CODEX/scRNA computational sources and direct utilities**. They preserve training, preprocessing and scoring implementations for methodological inspection. They are not the quick-start path and are not automatically executed. Step05 is retained only for historical upstream CN/NMF provenance. Shared utilities occur once; notebooks and staged working copies are not duplicated here. The original local Final/V9/Legacy sources remain independently protected.

Shared output helpers live in `reference_source/editable_pptx.py`: canvas checks, staged PNG/PPTX validation, paired-file promotion and review-Figure font configuration. Existing analysis modules import these functions under their original names. The training, marker-attribution and scRNA analysis modules remain separate; their calculations and process boundaries are unchanged. Output-path authorization remains in the original caller, and the separate legacy scRNA A4 export policy is preserved.

All source comments, docstrings and human-readable messages are provided in English. Functional identifiers, numeric settings, data keys and input/output paths retain their original values. The legacy layout-reference filename is written with Unicode escapes to preserve its exact runtime value. Translation does not constitute a new analysis run.

The previous `src/` plus `public_repository/source_templates/` duplication, auto-staging command framework and Opal analysis scripts have been removed from this publication tree. The retained reference code contains portable path tokens from the prior release and needs explicit workspace adaptation and external inputs before any research execution. Do not run these files as a drop-in workflow. See [computational references and environment](docs/scope.md).

Opal is limited to [packages used and request-based availability](docs/opal_packages.md); no Opal analysis code or data is included.

Repository: https://github.com/khsoon97/Analysis_script. See [citation status](CITATION.md). License and final manuscript citation require author confirmation. The repository access setting is managed separately from this code bundle.
