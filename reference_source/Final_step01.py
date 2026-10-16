# %%
from Final_utils import *

# ==============================================================================
# [Step 01-1] Check checkpoints and load data
# ==============================================================================
CHECKPOINT_STEP01 = os.path.join(CKPT_DIR, 'mv1_step01.h5ad')
STEP01_PREPROCESS_VERSION = '2026-05-15_exact_missing_label_v1'
STEP01_FIGURE_ONLY = True
FORCE_REBUILD_STEP01 = False

if STEP01_FIGURE_ONLY:
    if not os.path.exists(CHECKPOINT_STEP01):
        raise FileNotFoundError(f"STEP01_FIGURE_ONLY=True but the Step01 checkpoint is missing: {CHECKPOINT_STEP01}")
    print(f"🔹 [Step 01-1] Figure-only mode → load the existing checkpoint: {os.path.basename(CHECKPOINT_STEP01)}")
    # Figure 1B uses only ``obs`` metadata. Keep the 2.5 GB expression matrix
    # disk-backed so Figure-only export does not duplicate it in RAM.
    adata = sc.read_h5ad(CHECKPOINT_STEP01, backed='r')
    need_processing = False
elif FORCE_REBUILD_STEP01:
    print("🔹 [Step 01-1] FORCE_REBUILD_STEP01=True → ignore the checkpoint and reprocess.")
    need_processing = True
elif os.path.exists(CHECKPOINT_STEP01):
    print(f"🔹 [Step 01-1] Existing checkpoint found: loading {os.path.basename(CHECKPOINT_STEP01)}...")
    adata = sc.read_h5ad(CHECKPOINT_STEP01)
    cached_version = adata.uns.get('STEP01_PREPROCESS_VERSION')
    need_processing = cached_version != STEP01_PREPROCESS_VERSION
    if need_processing:
        print(
            "  ⚠️ Step01 preprocessing rules have changed; rebuilding the checkpoint. "
            f"(cached={cached_version}, expected={STEP01_PREPROCESS_VERSION})"
        )
        del adata
        gc.collect()
else:
    print("🔹 [Step 01-1] No checkpoint → start processing new data")
    need_processing = True

# %%
# ==============================================================================
# [Step 01-2 & 01-3] Clean data, remove debris, and map to standardized lineages
# ==============================================================================
if need_processing:
    if 'adata' not in dir():
        print("🔹 [Step 01-2] Loading metadata and removing debris ('na')...")

        annot_df = pd.read_csv(ANNOT_PATH, usecols=['unique_cell_id', 'x', 'y', 'unique_region', 'PatientID', 'Merged_final'])
        meta_df  = pd.read_csv(METADATA_PATH, usecols=['PatientID', 'Entity'])

        annot_df.rename(columns={'x': 'X_centroid', 'y': 'Y_centroid', 'unique_region': 'Run_Tissue_name'}, inplace=True)
        meta_df.rename(columns={'Entity': 'Disease'}, inplace=True)
        df_merged = pd.merge(annot_df, meta_df, on='PatientID', how='inner').set_index('unique_cell_id')

        label_raw = df_merged['Merged_final']
        label_clean = label_raw.astype('string').str.strip()
        na_mask = (
            label_raw.isna()
            | label_clean.eq('')
            | label_clean.str.fullmatch(r'(na|nan)', case=False, na=False)
        )
        if na_mask.any():
            print(f"  ⚠️ Exact-match missing/debris labels found: removing {na_mask.sum():,} cells.")
            df_merged = df_merged[~na_mask].copy()

        print("🔹 Loading H5AD and excluding nuclear markers (for QC)...")
        adata = sc.read_h5ad(H5AD_PATH)
        common_cells = adata.obs_names.intersection(df_merged.index)
        adata = adata[common_cells].copy()
        df_merged = df_merged.loc[common_cells]

        excluded_features = ['DAPI', 'Histone', 'Hoechst', 'DRAQ5', 'Blank', 'Empty']
        valid_features = [c for c in adata.var_names if not any(x in c for x in excluded_features)]
        adata = adata[:, valid_features].copy()
        print(f"  → Cleaning complete: {len(adata):,} cells / {len(valid_features)} features retained")

    print("🔹 [Step 01-3] Mapping to standardized lineages...")
    if 'df_merged' in dir():
        df_merged['GT_Subtype']       = df_merged['Merged_final'].apply(map_subtype)
        df_merged['GT_Lineage_Broad'] = df_merged['GT_Subtype'].apply(refine_broad_for_subtyping)
        adata.obs = df_merged

# %%
# ==============================================================================
# [Step 01-4] Rename Run_Tissue_name and visualize disease/sample data summaries
# ==============================================================================
print("🔹 [Step 01-4] Renaming Run_Tissue_name and generating visualizations...")

# Apply Run_Tissue_name renaming
if 'adata' in dir():
    adata.obs = rename_tissue_samples(adata.obs)
    print("  ✅ Tissue renaming complete. Example results:")
    sample_preview = adata.obs[['Disease', 'PatientID', 'Run_Tissue_name']].drop_duplicates().head(15)
    print(sample_preview.to_string(index=False))

df_obs = adata.obs.copy()
lineage_order = ['B cells', 'T cells', 'Myeloid cells', 'Stromal cells', 'NK cells']
lineage_colors_ordered = [LINEAGE_COLORS.get(l, '#AAAAAA') for l in lineage_order]

# Fix the disease order
df_obs['Disease'] = pd.Categorical(df_obs['Disease'], categories=DISEASE_ORDER, ordered=True)

# Summarize detailed counts
sample_detail_counts = df_obs.groupby(['Disease', 'Run_Tissue_name', 'GT_Lineage_Broad']).size().unstack(fill_value=0)
sample_detail_counts = sample_detail_counts[sample_detail_counts.sum(axis=1) > 0]
existing_cols = [c for c in lineage_order if c in sample_detail_counts.columns]
sample_detail_counts = sample_detail_counts[existing_cols]

print("\n📝 [Summary: Raw Cell Counts per Sample (Active Only)]")
with pd.option_context('display.max_rows', None, 'display.expand_frame_repr', False):
    print(sample_detail_counts.to_string())

total_active = len(sample_detail_counts)
avg_cells = sample_detail_counts.sum(axis=1).mean()
print(f"\n📊 Total Active Samples: {total_active}")
print(f"📊 Average Cells per Sample: {avg_cells:.1f}")

# ── Combined visualization (top: disease donut charts / bottom: vertical stacked bars for individual samples) ──
disease_counts = df_obs.groupby('Disease', observed=False).size()
disease_counts = disease_counts.reindex(DISEASE_ORDER).dropna()

cross_tab_dis = pd.crosstab(df_obs['Disease'], df_obs['GT_Lineage_Broad'])
cross_tab_dis = cross_tab_dis.reindex(DISEASE_ORDER).dropna()

df_plot = sample_detail_counts.copy()
df_plot_meta = pd.DataFrame(
    [{'Disease': disease, 'Sample': sample} for disease, sample in df_plot.index],
    index=[sample for _, sample in df_plot.index]
)
df_plot.index = df_plot_meta.index
df_plot_pct = df_plot.div(df_plot.sum(axis=1), axis=0) * 100

def annotate_pie_outside(ax, wedges, values, label_radius=1.10, connector_radius=0.88, min_pct=1.0):
    total = float(np.sum(values))
    if total <= 0:
        return

    for wedge, value in zip(wedges, values):
        pct = float(value) / total * 100.0
        if pct < min_pct:
            continue
        theta = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        x, y = np.cos(theta), np.sin(theta)
        label_x = label_radius * x
        label_y = label_radius * y
        ha = 'left' if x >= 0.10 else 'right' if x <= -0.10 else 'center'

        ax.annotate(
            f'{pct:.1f}%',
            xy=(connector_radius * x, connector_radius * y),
            xytext=(label_x, label_y),
            ha=ha, va='center',
            fontsize=10.2, fontweight='bold', color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.78, pad=0.04),
            arrowprops=dict(
                arrowstyle='-',
                color='0.25',
                lw=0.5,
                shrinkA=0,
                shrinkB=0,
                connectionstyle='arc3,rad=0.03'
            ),
            annotation_clip=False,
            clip_on=False
        )

num_diseases = len(disease_counts)
num_samples = len(df_plot_pct)
# Place donuts and stacked bars directly on a landscape canvas matching the edited version.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'pdf.use14corefonts': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})
# Preserve the slide-slot aspect ratio while increasing the native physical
# canvas so labels remain legible after the full Figure is placed in PPT.
figure_size = (15.20, 7.40)
fig = plt.figure(figsize=figure_size)
gs = fig.add_gridspec(2, 1, height_ratios=[0.66, 1.0], hspace=0.06)

# Use 1 unit between samples and an additional gap between diseases to clarify cohort boundaries.
disease_gap = 1.25
x_positions = []
current_x = 0.0
previous_disease = None
for disease in df_plot_meta['Disease']:
    if previous_disease is not None and disease != previous_disease:
        current_x += disease_gap
    x_positions.append(current_x)
    current_x += 1.0
    previous_disease = disease
x_positions = np.asarray(x_positions, dtype=float)
x_min = float(x_positions.min() - 0.65)
x_max = float(x_positions.max() + 0.65)

group_centers = {}
for disease in disease_counts.index:
    disease_mask = df_plot_meta['Disease'].astype(str).to_numpy() == str(disease)
    if disease_mask.any():
        group_centers[str(disease)] = float(np.mean(x_positions[disease_mask]))

# (A) Top: disease donut charts aligned with the centers of the sample groups below
ax_donut_panel = fig.add_subplot(gs[0, 0])
ax_donut_panel.set_xlim(x_min, x_max)
ax_donut_panel.set_ylim(0, 1)
ax_donut_panel.set_axis_off()
donut_width = min(0.185, 0.94 / max(num_diseases, 1))
donut_height = 0.90

for disease in disease_counts.index:
    disease_key = str(disease)
    if disease_key not in group_centers:
        continue

    x_frac = (group_centers[disease_key] - x_min) / (x_max - x_min)
    x_left = min(max(x_frac - donut_width / 2, 0.010), 0.990 - donut_width)
    ax_pie = ax_donut_panel.inset_axes([x_left, 0.015, donut_width, donut_height])

    data_full = cross_tab_dis.loc[disease, existing_cols].reindex(existing_cols).fillna(0)
    data = data_full[data_full > 0]
    colors = [LINEAGE_COLORS.get(l, '#AAAAAA') for l in data.index]

    wedges, _ = ax_pie.pie(
        data, labels=None, colors=colors, startangle=90, radius=1.0,
        wedgeprops=dict(width=0.40, edgecolor='white', linewidth=1.1)
    )
    annotate_pie_outside(ax_pie, wedges, data.to_numpy())
    ax_pie.text(
        0, 0,
        f"{int(disease_counts[disease]):,}\nCells",
        ha='center', va='center', fontsize=11.8, fontweight='bold'
    )
    ax_pie.set_xlim(-1.42, 1.42)
    ax_pie.set_ylim(-1.34, 1.34)
    ax_pie.set_axis_off()
    ax_pie.set_title(
        str(disease), fontsize=16.2, fontweight='bold', pad=1
    )

# (B) Bottom: vertical stacked bars by sample. The Y-axis shows percentages; samples are on the X-axis.
ax_bar = fig.add_subplot(gs[1, 0], sharex=ax_donut_panel)
bottom = np.zeros(num_samples, dtype=float)
for lineage, color in zip(lineage_order, lineage_colors_ordered):
    values = df_plot_pct[lineage].to_numpy() if lineage in df_plot_pct.columns else np.zeros(num_samples)
    ax_bar.bar(
        x_positions, values, bottom=bottom, width=0.82,
        color=color, edgecolor='white', linewidth=0.8, label=lineage
    )
    bottom += values

for idx in range(1, num_samples):
    if df_plot_meta['Disease'].iloc[idx] != df_plot_meta['Disease'].iloc[idx - 1]:
        ax_bar.axvline(
            (x_positions[idx] + x_positions[idx - 1]) / 2,
            color='0.72', linewidth=1.0, alpha=0.9
        )

ax_bar.set_xlabel('')
ax_bar.set_ylabel('Lineage proportion (%)', fontsize=17, fontweight='bold', labelpad=7)
ax_bar.set_xlim(x_min, x_max)
ax_bar.set_ylim(0, 100)
ax_bar.set_yticks([0, 25, 50, 75, 100])
ax_bar.set_xticks(x_positions)
ax_bar.set_xticklabels(df_plot_pct.index, rotation=90, ha='center', va='top')
ax_bar.tick_params(axis='x', labelsize=9, pad=1.5)
for t in ax_bar.get_xticklabels(): t.set_weight('bold')
ax_bar.tick_params(axis='y', labelsize=10)
for t in ax_bar.get_yticklabels(): t.set_weight('bold')

ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['left'].set_linewidth(1.2)
ax_bar.spines['bottom'].set_linewidth(1.2)
ax_bar.grid(axis='y', linestyle='--', alpha=0.42, zorder=0)

# (C) Keep a compact lineage key on the same lower band as the shared x label.
# Display-only abbreviations do not alter the underlying lineage categories.
handles, labels = ax_bar.get_legend_handles_labels()
legend = ax_bar.get_legend()
if legend is not None:
    legend.remove()
_lineage_legend_display = {
    'NK cells': 'NK',
    'Stromal cells': 'Stromal',
    'Myeloid cells': 'Myeloid',
    'T cells': 'T',
    'B cells': 'B',
}
fig.legend(
    handles[::-1], [_lineage_legend_display.get(label, label) for label in labels[::-1]],
    loc='center', bbox_to_anchor=(0.815, 0.030),
    ncol=5, fontsize=12.2,
    frameon=False, handlelength=1.20, handleheight=0.95,
    handletextpad=0.42, labelspacing=0.45, columnspacing=0.85,
    borderaxespad=0.0, borderpad=0.0
)
fig.text(0.475, 0.030, 'Sample', ha='center', va='center',
         fontsize=16.2, fontweight='bold')

# Reserve sufficient bottom margin when creating the figure so vertical sample labels
# and the X-axis title stay within the canvas; do not compensate with cropping/resizing during saving.
plt.subplots_adjust(left=0.0550, right=0.9920, top=0.978, bottom=0.2050)
fig_path = os.path.join(FIG_DIR, 'Final_mv1_Step01_Integrated_Summary_Vertical_Donuts_Final')
fig.set_dpi(300)
save_final_figure_pair(fig, f'{fig_path}.png', f'{fig_path}.pptx', dpi=300)
plt.close(fig)

# %%
# ==============================================================================
# [Step 01-5] Save the final Step 01 data
# ==============================================================================
if 'need_processing' in globals() and need_processing:
    print(f"\n📊 [Step 01] Final cleaned data summary")
    print(adata.obs['GT_Lineage_Broad'].value_counts())
    adata.uns['STEP01_PREPROCESS_VERSION'] = STEP01_PREPROCESS_VERSION
    adata.uns['STEP01_MISSING_LABEL_RULE'] = 'isna, blank string, or exact fullmatch: na|nan'
    adata.write_h5ad(CHECKPOINT_STEP01)
    print(f"\n✅ Step 01 saved: {CHECKPOINT_STEP01}")
else:
    print(f"\n✅ [Step 01] Checkpoint reused | Total cells: {len(adata):,}")
    if 'GT_Lineage_Broad' in adata.obs.columns:
        print(adata.obs['GT_Lineage_Broad'].value_counts().to_string())
