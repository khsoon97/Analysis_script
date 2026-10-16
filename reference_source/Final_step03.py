# %%
from Final_utils import *
from itertools import combinations as _combs
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, confusion_matrix
import rapids_singlecell as rsc

# ==============================================================================
# [Step 03-1] Broad Cell Type benchmark: load checkpoints
# ==============================================================================
CHECKPOINT_STEP02 = os.path.join(CKPT_DIR, 'mv1_step02.h5ad')
CHECKPOINT_STEP03 = os.path.join(CKPT_DIR, 'mv1_step03.h5ad')
MODELS_STEP02_PATH = os.path.join(CKPT_DIR, 'mv1_models_broad.pkl')
BENCH_STEP03_PATH = os.path.join(CKPT_DIR, 'mv1_step03_benchmark_results.csv')
CM_STEP03_DIR = os.path.join(CKPT_DIR, 'step03_confusion_matrices')
STEP03_FIGURE_ONLY = True
FORCE_RERUN_STEP03 = False
FORCE_RELOAD_STEP02_FOR_STEP03 = True  # Reload the Step02 checkpoint even if adata is in memory when True.
INCLUDE_SELECTED_BEFORE_RESCUE = False  # Exclude the selected combo before rescue from the representative Step03 figure because Step02 presents it.
INCLUDE_SELECTED_AFTER_RESCUE = True   # Include the selected combo after rescue in the representative figure/table.
USE_ML_LINEAGE_PURE_FOR_BEFORE_RESCUE = True  # Use ML_Lineage_Pure for Before Rescue when True.
DROP_SELECTED_COMBO_DUPLICATE = True   # Exclude duplicate rows for the selected combo from the list of 15 combos when True.
STEP03_ACTIVE_FIGURE_OUTPUTS = {
    'Final_mv1_Step03-06_Lineage_Performance_Compact.png',
    'Final_mv1_Step03-06_CM_Baseline_Selected.png',
    'Final_mv1_Step03-06_CM_Single_ML_Models.png',
    'Final_mv1_Step03-07_DLBCL_PerLineage_BAcc_F1.png',
}

def _clean_obs_string_step03(series):
    return series.astype('string').str.strip()

def make_exact_disease_mask_step03(adata_obj, disease_name):
    disease_clean = _clean_obs_string_step03(adata_obj.obs['Disease'])
    return disease_clean.eq(disease_name).fillna(False).to_numpy(dtype=bool)

def make_valid_label_mask_step03(values):
    label_clean = pd.Series(values).astype('string').str.strip()
    invalid_mask = (
        label_clean.isna()
        | label_clean.eq('')
        | label_clean.str.fullmatch(r'(unknown|na|nan)', case=False, na=False)
    )
    return (~invalid_mask).to_numpy(dtype=bool), label_clean

def _checkpoint_is_fresh(input_path, output_path):
    if not os.path.exists(output_path):
        return False
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Step03 input checkpoint is missing: {input_path}")
    return os.path.getmtime(output_path) >= os.path.getmtime(input_path)

def _safe_filename(label):
    safe = str(label).replace('+', 'plus').replace('/', '_').replace('\\', '_')
    safe = ''.join(ch if ch.isalnum() or ch in ('_', '-', '.') else '_' for ch in safe)
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip('_')[:120] or 'model'

if not os.path.exists(CHECKPOINT_STEP02):
    raise FileNotFoundError(f"Step03 requires completed Step02 results: {CHECKPOINT_STEP02}")

# First load the target_combo/combination information specified in Step 02.
step02_model_data = {}
step02_manifest = {}
target_combo = None
combo_names_eval = []
pred_combo_names = []
if os.path.exists(MODELS_STEP02_PATH):
    try:
        step02_model_data = joblib.load(MODELS_STEP02_PATH)
        step02_manifest = step02_model_data.get('selection_manifest', {})
        if step02_model_data.get('target_combo'):
            target_combo = step02_model_data['target_combo']
            print(f"🔹 Loaded the ensemble combination specified in Step 02: {target_combo}")
        elif step02_manifest.get('applied_combo'):
            target_combo = step02_manifest['applied_combo']
            print(f"🔹 Loaded the ensemble combination from the Step 02 manifest: {target_combo}")
        combo_names_eval = list(step02_model_data.get('combo_names', []))
    except Exception as e:
        print(f"⚠️ Failed to load Step02 model metadata: {e}")

if not combo_names_eval:
    broad_combos_eval = []
    for r in range(1, len(MODEL_KEYS) + 1):
        for combo in _combs(MODEL_KEYS, r):
            broad_combos_eval.append(combo)
    combo_names_eval = ['+'.join(c) for c in broad_combos_eval]

pred_combo_names = [f'ML ({cn})' for cn in combo_names_eval]

step03_fresh = _checkpoint_is_fresh(CHECKPOINT_STEP02, CHECKPOINT_STEP03)
if STEP03_FIGURE_ONLY:
    if not os.path.exists(CHECKPOINT_STEP03):
        raise FileNotFoundError(f"STEP03_FIGURE_ONLY=True but the Step03 checkpoint is missing: {CHECKPOINT_STEP03}")
    if not os.path.exists(BENCH_STEP03_PATH):
        raise FileNotFoundError(f"STEP03_FIGURE_ONLY=True but the Step03 benchmark table is missing: {BENCH_STEP03_PATH}")
    print(f"🔹 [Step 03] Figure-only mode → load saved completed Step03 results: {os.path.basename(CHECKPOINT_STEP03)}")
    step03_skip = True
    adata_bench = sc.read_h5ad(CHECKPOINT_STEP03, backed='r')
    print(f"🔹 Step 03 checkpoint loaded: {len(adata_bench):,} cells")
elif os.path.exists(CHECKPOINT_STEP03) and step03_fresh and not FORCE_RERUN_STEP03:
    print(f"✅ Up-to-date Step 03 checkpoint found → skip computation: {os.path.basename(CHECKPOINT_STEP03)}")
    step03_skip = True
    adata_bench = sc.read_h5ad(CHECKPOINT_STEP03)
    print(f"🔹 Step 03 checkpoint loaded: {len(adata_bench):,} cells")
else:
    if os.path.exists(CHECKPOINT_STEP03) and not step03_fresh:
        print("⚠️ The Step 02 checkpoint is newer than Step 03. Recalculating Step 03.")
    if FORCE_RERUN_STEP03:
        print("🔹 FORCE_RERUN_STEP03=True → recalculate Step 03.")
    step03_skip = False
    if FORCE_RELOAD_STEP02_FOR_STEP03 or 'adata' not in globals():
        adata = sc.read_h5ad(CHECKPOINT_STEP02)
        print(f"🔹 Load Step 02 checkpoint: {os.path.basename(CHECKPOINT_STEP02)}")
    else:
        print("🔹 Using adata in memory. Set FORCE_RELOAD_STEP02_FOR_STEP03=True to force loading the latest Step02 h5ad.")

    required_cols = ['Disease', 'GT_Lineage_Broad', 'ML_Lineage_Raw']
    missing_required = [c for c in required_cols if c not in adata.obs.columns]
    if missing_required:
        raise KeyError(f"Required Step03 obs columns are missing: {missing_required}")

    dlbcl_mask = make_exact_disease_mask_step03(adata, 'DLBCL')
    valid_gt_mask, _ = make_valid_label_mask_step03(adata.obs['GT_Lineage_Broad'].values)
    adata_bench = adata[dlbcl_mask & valid_gt_mask].copy()
    print(f"  → Created a new benchmark dataset: {len(adata_bench):,} cells")

if 'adata_bench' in locals():
    missing_pred_cols = [f'Pred_ML_Lineage_{i}' for i in range(1, len(combo_names_eval) + 1)
                         if f'Pred_ML_Lineage_{i}' not in adata_bench.obs.columns]
    if missing_pred_cols:
        raise KeyError(f"Step02 combination prediction columns required by Step03 are missing: {missing_pred_cols[:5]}")
    if step03_skip and os.path.exists(BENCH_STEP03_PATH):
        df_bench = pd.read_csv(BENCH_STEP03_PATH)
        if INCLUDE_SELECTED_BEFORE_RESCUE and target_combo and target_combo in combo_names_eval:
            target_before_name = f"ML ({target_combo}) Before Rescue"
        print(f"🔹 Restored the Step 03 benchmark table: {BENCH_STEP03_PATH}")

# %%
# ==============================================================================
# [Step 03-2] Clustering-based prediction (Leiden, Default Parameters)
# ==============================================================================
if not step03_skip:
    print("🔹 [Step 03-2] Performing Leiden clustering-based annotation...")

    rsc.pp.pca(adata_bench, n_comps=30)
    rsc.pp.neighbors(adata_bench, n_neighbors=15, use_rep='X_pca')
    rsc.tl.leiden(adata_bench, resolution=1.0, key_added='leiden_default')

    y_true = adata_bench.obs['GT_Lineage_Broad']
    cluster_map = {}
    for cluster in adata_bench.obs['leiden_default'].unique():
        subset_gt = adata_bench.obs.loc[adata_bench.obs['leiden_default'] == cluster, 'GT_Lineage_Broad']
        cluster_map[cluster] = subset_gt.value_counts().idxmax()

    adata_bench.obs['Pred_Clustering'] = adata_bench.obs['leiden_default'].map(cluster_map).values
    score = balanced_accuracy_score(y_true, adata_bench.obs['Pred_Clustering'])
    print(f"  ✅ Clustering complete (Resolution: 1.0 / Balanced Acc: {score*100:.2f}%)")

# %%
# ==============================================================================
# [Step 03-3] SC-Centroid-based prediction (GPU batch processing)
# ==============================================================================
if not step03_skip:
    print("🔹 [Step 03-3] Performing SC-Centroid-based prediction...")

    ref_mask_b = ~make_exact_disease_mask_step03(adata, 'DLBCL')
    valid_ref_mask_b, _ = make_valid_label_mask_step03(adata.obs['GT_Lineage_Broad'].values)
    adata_ref_tmp = adata[ref_mask_b & valid_ref_mask_b]

    classes = sorted(adata_ref_tmp.obs['GT_Lineage_Broad'].unique())
    centroids = []
    for cls in classes:
        cls_data = adata_ref_tmp[adata_ref_tmp.obs['GT_Lineage_Broad'] == cls].X
        if hasattr(cls_data, 'toarray'):
            centroids.append(np.array(cls_data.mean(axis=0)).flatten())
        else:
            centroids.append(np.mean(cls_data, axis=0))

    centroids = np.vstack(centroids).astype(np.float32)
    centroids_gpu = cp.array(centroids)
    norm_C = cp.linalg.norm(centroids_gpu, axis=1, keepdims=True)
    norm_C[norm_C == 0] = 1.0

    X_bench_raw = adata_bench.X.toarray() if hasattr(adata_bench.X, 'toarray') else adata_bench.X
    n_total = X_bench_raw.shape[0]
    chunk_size = 200000
    all_pred_indices = []

    for s in tqdm(range(0, n_total, chunk_size), desc="Centroid Prediction"):
        e = min(s + chunk_size, n_total)
        X_chunk_gpu = cp.array(X_bench_raw[s:e].astype(np.float32))
        norm_X = cp.linalg.norm(X_chunk_gpu, axis=1, keepdims=True)
        norm_X[norm_X == 0] = 1.0
        sim_chunk = (X_chunk_gpu @ centroids_gpu.T) / (norm_X @ norm_C.T)
        all_pred_indices.append(cp.argmax(sim_chunk, axis=1).get())
        del X_chunk_gpu, norm_X, sim_chunk
        cp.get_default_memory_pool().free_all_blocks()

    pred_idx_final = np.concatenate(all_pred_indices)
    adata_bench.obs['Pred_SC_Centroid'] = [classes[i] for i in pred_idx_final]
    print(f"  → Predictions complete for {n_total:,} cells in total")
    del centroids_gpu, norm_C; gc.collect(); cp.get_default_memory_pool().free_all_blocks()

# %%
# ==============================================================================
# [Step 03-4] Compare model performance and automatically select the best combination
# ==============================================================================
if 'adata_bench' in locals():
    print("🔹 [Step 03-4] Comparing model performance and selecting the best combination...")

    if step02_manifest:
        adata_bench.uns['step02_model_selection'] = step02_manifest

    models = {
        'Leiden + GT-majority': 'Pred_Clustering',
        'SC-Centroid': 'Pred_SC_Centroid'
    }
    
    target_before_name = None
    if INCLUDE_SELECTED_BEFORE_RESCUE and target_combo and target_combo in combo_names_eval:
        pred_target_col = f"Pred_ML_Lineage_{combo_names_eval.index(target_combo) + 1}"
        target_col = 'ML_Lineage_Pure' if USE_ML_LINEAGE_PURE_FOR_BEFORE_RESCUE and 'ML_Lineage_Pure' in adata_bench.obs.columns else pred_target_col
        target_before_name = f"ML ({target_combo}) Before Rescue"
        models[target_before_name] = target_col

    # ── [Fix] Add the final Step 02 results (including rescue) to the benchmark ──
    if INCLUDE_SELECTED_AFTER_RESCUE and 'ML_Lineage_Raw' in adata_bench.obs.columns:
        rescued_name = f"ML ({target_combo}) + Rescue" if 'target_combo' in locals() and target_combo else "ML (Final) + Rescue"
        models[rescued_name] = 'ML_Lineage_Raw'

    for i, cn in enumerate(combo_names_eval, 1):
        if DROP_SELECTED_COMBO_DUPLICATE and target_combo and cn == target_combo and target_before_name:
            continue
        model_name = f'ML ({cn})'
        if model_name not in models:
            models[model_name] = f'Pred_ML_Lineage_{i}'

    y_true = adata_bench.obs['GT_Lineage_Broad']
    ordered_classes = ['B cells', 'Myeloid cells', 'NK cells', 'Stromal cells', 'T cells']
    ordered_classes = [c for c in ordered_classes if c in y_true.unique()]
    if step03_skip and 'df_bench' in locals() and not df_bench.empty:
        if 'Avg_Score' not in df_bench.columns:
            df_bench['Avg_Score'] = df_bench[['Balanced Acc', 'Macro F1', 'Weighted F1']].mean(axis=1)
        print("  ⏩ Figure-only mode: use the saved benchmark table and skip metric recalculation.")
    else:
        bench_results = []
        for name, col in models.items():
            if col not in adata_bench.obs.columns: continue
            y_pred = adata_bench.obs[col].astype(str)
            y_t = y_true.astype(str)
            existing = np.unique(y_t)
            b_acc = balanced_accuracy_score(y_t, y_pred)
            macro_f1 = f1_score(y_t, y_pred, average='macro', zero_division=0, labels=existing)
            weighted_f1 = f1_score(y_t, y_pred, average='weighted', zero_division=0)
            acc = accuracy_score(y_t, y_pred)
            bench_results.append({
                'Model': name, 'Column': col, 'Balanced Acc': b_acc,
                'Macro F1': macro_f1, 'Weighted F1': weighted_f1, 'Acc': acc,
                'N': int(len(y_t))
            })

        df_bench = pd.DataFrame(bench_results)

        if not df_bench.empty:
            df_bench['Avg_Score'] = df_bench[['Balanced Acc', 'Macro F1', 'Weighted F1']].mean(axis=1)

            # ── 1) Top 5 ranking by Balanced Acc ──
            print("\n🥇 [Top 5 by Balanced Accuracy]")
            df_bacc = df_bench.sort_values(by='Balanced Acc', ascending=False).head(5)
            for rank, (_, row) in enumerate(df_bacc.iterrows(), 1):
                print(f"  Rank {rank:2d}: {row['Model']:<25} | B-Acc: {row['Balanced Acc']*100:.2f}%")

            # ── 2) Top 5 ranking by Macro F1 ──
            print("\n🥇 [Top 5 by Macro F1-Score]")
            df_f1 = df_bench.sort_values(by='Macro F1', ascending=False).head(5)
            for rank, (_, row) in enumerate(df_f1.iterrows(), 1):
                print(f"  Rank {rank:2d}: {row['Model']:<25} | F1: {row['Macro F1']*100:.2f}%")

            # ── 3) Top 10 ranking by overall average (Avg) ──
            print("\n🏆 [Top 10 by Overall Evaluation (Avg Score)]")
            df_sorted = df_bench.sort_values(by='Avg_Score', ascending=False).head(10)
            for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
                print(f"  Rank {rank:2d}: {row['Model']:<25} | Avg: {row['Avg_Score']*100:.2f}%")

            # Automatically select the best model
            best_model_row = df_bench.loc[df_bench['Avg_Score'].idxmax()]
            print(f"\n⭐ Best overall final model: {best_model_row['Model']} (Avg: {best_model_row['Avg_Score']*100:.2f}%)")

            os.makedirs(CM_STEP03_DIR, exist_ok=True)
            df_bench.to_csv(BENCH_STEP03_PATH, index=False)
            print(f"  ✅ Benchmark table saved: {BENCH_STEP03_PATH}")

            for _, row in df_bench.iterrows():
                model_name = row['Model']
                col = row['Column']
                if col not in adata_bench.obs.columns:
                    continue
                cm_count = pd.crosstab(y_true.astype(str), adata_bench.obs[col].astype(str))
                cm_count = cm_count.reindex(index=ordered_classes, columns=ordered_classes, fill_value=0)
                cm_norm = cm_count.div(cm_count.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
                base_name = _safe_filename(model_name)
                cm_count.to_csv(os.path.join(CM_STEP03_DIR, f'{base_name}_counts.csv'))
                cm_norm.to_csv(os.path.join(CM_STEP03_DIR, f'{base_name}_row_norm.csv'))

# %%
# ==============================================================================
# [Step 03-5] Save Step 03 benchmark results (before visualization)
# ==============================================================================
if not step03_skip:
    adata_bench.write_h5ad(CHECKPOINT_STEP03)
    print(f"✅ Step 03 benchmark analysis results saved: {CHECKPOINT_STEP03}")

# %%
# ==============================================================================
# [Step 03-6] Visualize benchmark performance (after saving)
# ==============================================================================
# Specify a particular model name to force its inclusion in the plot (e.g., 'ML (XGB+MLP) + Rescue').
# Leave blank ('') to automatically select the rescued Step 02 combination or the overall top model, as before.
MANUAL_PLOT_MODEL = ''
# Use A4 only as the final assembly boundary; apply the physical aspect ratio of
# the edited version directly to each individual figure canvas.
# Design canvas dimensions to include rotated model names and panel titles before saving.
STEP03_BAR_FIGSIZE = (12.80, 3.70)
STEP03_CM_ROW_FIGSIZE = (663.148 / 72, 247.718 / 72)
STEP03_CM_GRID_FIGSIZE = (539.872 / 72, 7.20)
STEP03_PER_LINEAGE_FIGSIZE = (11.25, 3.65)

def _save_step03_png_pptx(fig, png_path, dpi=300):
    """Save the generated compact canvas as both PNG and PPTX without changing it."""
    if os.path.basename(png_path) not in STEP03_ACTIVE_FIGURE_OUTPUTS:
        raise PermissionError(f'Save path is outside the Step03 active Figure set: {png_path}')
    pptx_path = os.path.splitext(png_path)[0] + '.pptx'
    saved_png_path, saved_pptx_path = save_final_figure_pair(
        fig, png_path, pptx_path,
        dpi=dpi, facecolor='white', validate=True,
        show=False, close=False,
    )
    print(f"    ✅ PNG saved: {saved_png_path}")
    print(f"    ✅ PPTX saved: {saved_pptx_path}")
    return saved_png_path

if 'df_bench' in locals() and not df_bench.empty:
    print("🔹 [Step 03-6] Visualizing benchmark performance...")

    if 'df_bench' in locals() and not df_bench.empty:
        # Find the overall top model
        best_model_name = df_bench.loc[df_bench['Avg_Score'].idxmax(), 'Model']
        
        # 1. Prefer the manually specified model if it exists
        if MANUAL_PLOT_MODEL and MANUAL_PLOT_MODEL in df_bench['Model'].values:
            manual_model_name = MANUAL_PLOT_MODEL
            print(f"  👉 Applied the manually specified model to the plot: {manual_model_name}")
        else:
            # 2. Automatic logic: use the manual target combo specified in Step 02-3.5
            # (Prefer the rescued model added in Step 03-4 if available.)
            t_combo_str = target_combo if 'target_combo' in globals() and target_combo else None
            rescued_opt = f"ML ({t_combo_str}) + Rescue" if t_combo_str else "ML (Final) + Rescue"
            
            if rescued_opt in df_bench['Model'].values:
                manual_model_name = rescued_opt
            else:
                manual_model_name = f"ML ({t_combo_str})" if t_combo_str else best_model_name
        
        def _display_model_label_step03(model_name):
            """Display the final rescued model name concisely in the manuscript figure."""
            return str(model_name).replace(' + Rescue', '')

        def _display_model_label_step03_bar(model_name):
            """Wrap long baseline labels into two lines only in the compact bar figure."""
            display_name = _display_model_label_step03(model_name)
            if display_name == 'Leiden + GT-majority':
                return 'Leiden +\nGT-majority'
            if display_name == 'ML (XGB+MLP)':
                return 'ML\n(XGB+MLP)'
            return display_name

        # Compact manuscript bar chart (2 baselines + 4 individual models + the selected final ensemble)
        repr_models = [
            'Leiden + GT-majority', 'SC-Centroid',
            'ML (RF)', 'ML (XGB)', 'ML (ENet)', 'ML (MLP)'
        ]
        
        # Force the user-selected model to the end
        if manual_model_name not in repr_models:
            repr_models.append(manual_model_name)

        df_repr = df_bench[df_bench['Model'].isin(repr_models)].copy()
        df_repr['Model_Display'] = df_repr['Model'].map(_display_model_label_step03_bar)
        repr_model_labels = [_display_model_label_step03_bar(m) for m in repr_models]

        if not df_repr.empty:
            comp_palette = {
                'Leiden + GT-majority': '#BDC3C7', 'SC-Centroid': '#5DADE2',
                'ML (RF)': '#F1C40F', 'ML (XGB)': '#E67E22',
                'ML (ENet)': '#E74C3C', 'ML (MLP)': '#C0392B'
            }
            if 'target_before_name' in globals() and target_before_name:
                comp_palette[target_before_name] = '#7F8C8D'
            if manual_model_name not in comp_palette:
                comp_palette[manual_model_name] = '#943126'
            comp_palette_display = {
                _display_model_label_step03_bar(k): v
                for k, v in comp_palette.items()
                if _display_model_label_step03_bar(k) in repr_model_labels
            }
                
            fig, axes = plt.subplots(1, 3, figsize=STEP03_BAR_FIGSIZE, sharey=True)
            for idx, (met_col, met_title) in enumerate([
                ('Balanced Acc', 'Balanced Accuracy'),
                ('Macro F1', 'Macro F1-Score'),
                ('Weighted F1', 'Weighted F1-Score')
            ]):
                sns.barplot(data=df_repr, x='Model_Display', y=met_col, palette=comp_palette_display,
                            edgecolor='black', ax=axes[idx], order=repr_model_labels, linewidth=0.9)
                axes[idx].set_ylim(0, 1.10)
                axes[idx].set_yticks(np.arange(0.0, 1.01, 0.2))
                axes[idx].set_title(met_title, fontsize=16.8, fontweight='bold', pad=4)
                axes[idx].set_ylabel('')
                axes[idx].set_xlabel('')
                axes[idx].tick_params(axis='y', labelsize=11.8)
                if idx > 0:
                    axes[idx].tick_params(axis='y', labelleft=False)
                plt.setp(axes[idx].get_xticklabels(), fontweight='bold', fontsize=11.8, rotation=46, ha='right', rotation_mode='anchor')
                for tick_label in axes[idx].get_xticklabels():
                    tick_label.set_multialignment('center')
                plt.setp(axes[idx].get_yticklabels(), fontweight='bold')
                _visible_patches = [p for p in axes[idx].patches if p.get_height() > 0]
                for patch_idx, p in enumerate(_visible_patches):
                    h = p.get_height()
                    _label_offset = 4
                    axes[idx].annotate(f"{h*100:.1f}%",
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', xytext=(0, _label_offset),
                        textcoords='offset points', fontsize=11.8, fontweight='bold')
            fig.supylabel('Score', x=0.010, fontsize=15.5, fontweight='bold')
            fig.subplots_adjust(left=0.055, right=0.994, bottom=0.300, top=0.865, wspace=0.050)
            bar_path = os.path.join(FIG_DIR, 'Final_mv1_Step03-06_Lineage_Performance_Compact.png')
            _save_step03_png_pptx(fig, bar_path)
            plt.show()

        # Confusion matrices (representative models): export baselines/selected ensemble separately from individual ML models, as requested.
        def _wrap_step03_lineage_tick(label):
            """Wrap only the two long lineage names used as heatmap ticks."""
            return {
                'Myeloid cells': 'Myeloid\ncells',
                'Stromal cells': 'Stromal\ncells',
            }.get(str(label), str(label))

        def _plot_step03_cm_group(model_names, output_filename, group_title, ncols=None):
            group_items = [
                (name, models[name])
                for name in model_names
                if name in models and models[name] in adata_bench.obs.columns
            ]
            n_group = len(group_items)
            if n_group == 0:
                print(f"    ⚠️ No models are available to display in the confusion matrix group: {group_title}")
                return

            # Place one shared color bar in a dedicated axis on the right of the figure for the common scale.
            # Placing one beside each heatmap disrupts panel size and alignment on the compact direct canvas.
            ncols_eff = min(ncols or 2, n_group)
            nrows_eff = (n_group + ncols_eff - 1) // ncols_eff
            is_single_row = nrows_eff == 1
            fig_size = STEP03_CM_ROW_FIGSIZE if is_single_row else STEP03_CM_GRID_FIGSIZE
            fig = plt.figure(figsize=fig_size)
            if is_single_row:
                gs = fig.add_gridspec(
                    nrows_eff, ncols_eff,
                    left=0.082, right=0.905, bottom=0.24, top=0.91,
                    wspace=0.065, hspace=0.0,
                )
                # Reserve a distinct lane between the third heatmap and the
                # vertical colorbar so its horizontal title cannot collide
                # with the last annotated cell.
                cbar_rect = [0.918, 0.405, 0.0075, 0.255]
                title_y = 0.985
            else:
                gs = fig.add_gridspec(
                    nrows_eff, ncols_eff,
                    left=0.105, right=0.910, bottom=0.16, top=0.950,
                    wspace=0.075, hspace=0.13,
                )
                cbar_rect = [0.919, 0.405, 0.0075, 0.235]
                title_y = 0.975
            axes_flat = [
                fig.add_subplot(gs[row_idx, col_idx])
                for row_idx in range(nrows_eff)
                for col_idx in range(ncols_eff)
            ]
            cbar_ax = fig.add_axes(cbar_rect)

            for idx_cm, (name, col) in enumerate(group_items):
                ax = axes_flat[idx_cm]
                row_idx = idx_cm // ncols_eff
                col_idx = idx_cm % ncols_eff
                cm = pd.crosstab(y_true, adata_bench.obs[col], normalize='index')
                cm = cm.reindex(index=ordered_classes, columns=ordered_classes, fill_value=0)
                draw_cbar = idx_cm == 0
                sns.heatmap(cm, annot=True, fmt='.2f', cmap='RdPu', ax=ax,
                            cbar=draw_cbar, vmin=0, vmax=1.0, square=True,
                            cbar_ax=cbar_ax if draw_cbar else None,
                            annot_kws={'fontsize': 9.5, 'fontweight': 'bold'},
                            cbar_kws={'orientation': 'vertical'})
                ax.set_xticklabels([
                    _wrap_step03_lineage_tick(t.get_text())
                    for t in ax.get_xticklabels()
                ])
                ax.set_yticklabels([
                    _wrap_step03_lineage_tick(t.get_text())
                    for t in ax.get_yticklabels()
                ])
                title_prefix = "* " if name == selected_cm_model_name else ""
                ax.set_title(f"{title_prefix}{_display_model_label_step03(name)}", fontsize=12.5, fontweight='bold', pad=7)
                show_x_axis = is_single_row or row_idx == nrows_eff - 1
                show_y_axis = col_idx == 0
                ax.set_xlabel('', fontsize=11, fontweight='bold')
                ax.set_ylabel("Ground Truth" if show_y_axis else "", fontsize=11, fontweight='bold')
                ax.set_aspect('equal', adjustable='box')
                if show_x_axis:
                    plt.setp(ax.get_xticklabels(), fontweight='bold', rotation=42, ha='right', rotation_mode='anchor', fontsize=9)
                else:
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', length=0)
                if show_y_axis:
                    plt.setp(ax.get_yticklabels(), fontweight='bold', rotation=0, fontsize=9.5)
                else:
                    ax.set_yticklabels([])
                    ax.tick_params(axis='y', length=0)

            for j in range(n_group, len(axes_flat)):
                axes_flat[j].axis('off')

            cbar_ax.tick_params(axis='y', labelsize=9, pad=2)
            cbar_ax.set_ylabel('')
            # Each ground-truth row is normalized independently.  A compact
            # two-line title keeps the scientifically precise label centered
            # over the narrow bar without covering the final heatmap cell.
            cbar_ax.set_title('Row\nfraction', fontsize=9.4, fontweight='bold', pad=3, loc='center', x=0.5)
            fig.supxlabel('Predicted', fontsize=11.5, fontweight='bold', y=0.025)
            cm_path = os.path.join(FIG_DIR, output_filename)
            _save_step03_png_pptx(fig, cm_path)
            plt.show()

        selected_cm_model_name = 'ML (XGB+MLP) + Rescue'
        if selected_cm_model_name not in models:
            selected_cm_model_name = 'ML (XGB+MLP)' if 'ML (XGB+MLP)' in models else manual_model_name

        cm_primary_models = ['Leiden + GT-majority', 'SC-Centroid', selected_cm_model_name]
        cm_single_ml_models = ['ML (RF)', 'ML (XGB)', 'ML (ENet)', 'ML (MLP)']

        _plot_step03_cm_group(
            cm_primary_models,
            'Final_mv1_Step03-06_CM_Baseline_Selected.png',
            'Confusion Matrix (DLBCL Test Set): Baselines and Selected Ensemble',
            ncols=3,
        )
        _plot_step03_cm_group(
            cm_single_ml_models,
            'Final_mv1_Step03-06_CM_Single_ML_Models.png',
            'Confusion Matrix (DLBCL Test Set): Single ML Models',
            ncols=2,
        )

# %%
# ==============================================================================
# [Step 03-7] Lineage-wise performance heatmap for the DLBCL test set
# ==============================================================================
# Overall average Step03 performance alone makes it difficult to identify whether a particular lineage lowers performance.
# Present one-vs-rest BAcc/F1 for each representative model to check cell-type-specific
# prediction stability in advance of possible reviewer requests.
PLOT_STEP03_PER_LINEAGE_HEATMAP = True

if PLOT_STEP03_PER_LINEAGE_HEATMAP and 'adata_bench' in locals() and 'models' in locals():
    print("🔹 [Step 03-7] Generating lineage-wise BAcc/F1 heatmaps for the DLBCL test set...")

    def _display_model_label_step03_per_lineage(model_name):
        """Display the final rescued model name concisely in the manuscript figure."""
        display_name = str(model_name).replace(' + Rescue', '')
        if display_name == 'Leiden + GT-majority':
            return 'Leiden +\nGT-majority'
        if display_name == 'ML (XGB+MLP)':
            return 'ML\n(XGB+MLP)'
        return display_name

    if 'repr_models' in locals() and repr_models:
        per_lineage_model_names = [m for m in repr_models if m in models and models[m] in adata_bench.obs.columns]
    else:
        per_lineage_model_names = [m for m, col in models.items() if col in adata_bench.obs.columns]

    y_true_lineage = adata_bench.obs['GT_Lineage_Broad'].astype(str)
    if 'ordered_classes' not in locals() or not ordered_classes:
        ordered_classes = ['B cells', 'Myeloid cells', 'NK cells', 'Stromal cells', 'T cells']
        ordered_classes = [c for c in ordered_classes if c in y_true_lineage.unique()]

    per_lineage_rows = []
    for model_name in per_lineage_model_names:
        pred_col = models[model_name]
        y_pred_lineage = adata_bench.obs[pred_col].astype(str)
        for lineage_name in ordered_classes:
            true_bin = y_true_lineage.eq(lineage_name).to_numpy()
            pred_bin = y_pred_lineage.eq(lineage_name).to_numpy()
            tn, fp, fn, tp = confusion_matrix(true_bin, pred_bin, labels=[False, True]).ravel()

            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            f1_cls = (
                2 * precision * recall / (precision + recall)
                if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
                else 0.0
            )
            bacc_cls = np.nanmean([recall, specificity])

            per_lineage_rows.append({
                'Model': model_name,
                'Model_Display': _display_model_label_step03_per_lineage(model_name),
                'Lineage': lineage_name,
                'BAcc': bacc_cls,
                'F1': f1_cls,
                'Recall': recall,
                'Specificity': specificity,
                'Precision': precision,
                'Support': int(tp + fn),
                'Predicted_N': int(tp + fp),
            })

    per_lineage_df = pd.DataFrame(per_lineage_rows)
    if per_lineage_df.empty:
        print("    ⚠️ No representative models are available for lineage-wise performance calculation.")
    else:
        model_display_order = [
            _display_model_label_step03_per_lineage(m)
            for m in per_lineage_model_names
            if _display_model_label_step03_per_lineage(m) in per_lineage_df['Model_Display'].values
        ]
        fig = plt.figure(figsize=STEP03_PER_LINEAGE_FIGSIZE)
        gs = fig.add_gridspec(1, 2, left=0.102, right=0.910, bottom=0.275, top=0.90, wspace=0.030)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        # Keep a vertical color scale while separating its horizontal title
        # from the 100 tick label in the shallow manuscript slot.
        cbar_ax = fig.add_axes([0.919, 0.405, 0.0075, 0.235])
        heatmap_specs = [
            ('BAcc', 'One-vs-rest Balanced Accuracy'),
            ('F1', 'One-vs-rest F1-Score'),
        ]
        for idx_hm, (ax, (metric_col, metric_title)) in enumerate(zip(axes, heatmap_specs)):
            mat = per_lineage_df.pivot(index='Lineage', columns='Model_Display', values=metric_col)
            mat = mat.reindex(index=ordered_classes, columns=model_display_order)
            draw_cbar = idx_hm == len(heatmap_specs) - 1
            sns.heatmap(
                mat * 100,
                annot=True,
                fmt='.1f',
                cmap='YlGnBu',
                vmin=0,
                vmax=100,
                linewidths=0.8,
                linecolor='white',
                cbar=draw_cbar,
                cbar_ax=cbar_ax if draw_cbar else None,
                cbar_kws={'orientation': 'vertical'},
                annot_kws={'fontsize': 8.8, 'fontweight': 'bold'},
                ax=ax,
            )
            ax.set_title(metric_title, fontsize=12.5, fontweight='bold', pad=7)
            ax.set_xlabel('')
            ax.set_ylabel('Ground truth broad lineage' if idx_hm == 0 else '', fontsize=10.5, fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=8.2, fontweight='bold')
            for tick_label in ax.get_xticklabels():
                tick_label.set_multialignment('center')
            if idx_hm == 0:
                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9.2, fontweight='bold')
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)
        cbar_ax.tick_params(axis='y', labelsize=9, pad=2)
        cbar_ax.set_ylabel('')
        cbar_ax.set_title('Score (%)', fontsize=9.3, fontweight='bold', pad=7, loc='center', x=0.5)

        per_lineage_png = os.path.join(FIG_DIR, 'Final_mv1_Step03-07_DLBCL_PerLineage_BAcc_F1.png')
        _save_step03_png_pptx(fig, per_lineage_png)
        plt.show()
