# ==============================================================================
# [Step 05] Cellular Neighborhood (CN) Analysis
# ▷ Purpose: calculate subtype proportions among each cell's spatial neighbors (K-NN)
#             to derive Cellular Neighborhood clusters within the tumor microenvironment (TME)
# ▷ Input: ML_Subtype_Raw (or ML_Subtype_Pure) generated in Step 04
# ▷ Output: assign CN labels to cells and save CN/NMF spatial maps and CN-vs-NMF comparison figures
# ▷ Interpretation: treat NMF as cellular neighborhood composition topics, rather than subtypes.
# ==============================================================================

# %%
# ==============================================================================
# [Step 05-1] Cellular Neighborhood (CN) analysis: load checkpoints
# ▷ Purpose: use Step 04 results to calculate subtype proportions among spatial neighbors (K-NN)
#             to derive Cellular Neighborhood clusters within the tumor microenvironment (TME)
# ▷ Input: ML_Subtype_Raw (or ML_Subtype_Pure) generated in Step 04
# ▷ Output: assign CN labels to cells and save CN/NMF spatial maps and CN-vs-NMF comparison figures
# ==============================================================================
import time  # Prevent NameError after a kernel restart
from cuml.cluster import KMeans as cuKMeans
import colorsys
from matplotlib.colors import to_hex
from Final_utils import *
import joblib

# Incorporate meeting notes: NMF topics complement CN composition decomposition and are not subtype labels.
STEP05_NMF_K_LIST = [6, 8, 10]
STEP05_COMPARE_CN_NMF = True
STEP05_CN_NMF_COMPARE_K_LIST = [6, 8, 10]
STEP05_DIFF_MARKER_PADJ_ALPHA = 0.05
STEP05_DIFF_MARKER_PADJ_METHOD = 'fdr_bh'
STEP05_NMF_TOPIC_NOTE = 'NMF topics represent cellular-neighborhood composition patterns, not cell subtypes.'
STEP05_NMF_SPATIAL_LEGEND_TITLE = 'Color index: NMF topic (dominant neighborhood, cell %)'
STEP05_FIGURE_ONLY = True
# Define paths
CHECKPOINT_STEP04 = os.path.join(CKPT_DIR, 'mv1_step04B_pan_cell.h5ad') # Use only Step04B as the preceding V9 step
CHECKPOINT_STEP05 = os.path.join(CKPT_DIR, 'mv1_step05_cn.h5ad')       # Current step
if STEP05_FIGURE_ONLY:
    if not os.path.exists(CHECKPOINT_STEP05):
        raise FileNotFoundError(f"STEP05_FIGURE_ONLY=True but the Step05 checkpoint is missing: {CHECKPOINT_STEP05}")
    print(f"🔹 [Step 05] Figure-only mode → load the existing CN/NMF checkpoint: {os.path.basename(CHECKPOINT_STEP05)}")
    adata = sc.read_h5ad(CHECKPOINT_STEP05)
    step05_skip = True
elif os.path.exists(CHECKPOINT_STEP05):
    # First priority: CN analysis is already complete (load this step's own checkpoint)
    print(f"✅ [Step 05] Load the completed checkpoint: {os.path.basename(CHECKPOINT_STEP05)}")
    adata = sc.read_h5ad(CHECKPOINT_STEP05)
    step05_skip = True
elif os.path.exists(CHECKPOINT_STEP04):
    # Second priority: Step 04 is complete and Step 05 needs to start
    print(f"🔹 [Step 05] Load the previous step (Step 04) results → prepare CN analysis: {os.path.basename(CHECKPOINT_STEP04)}")
    adata = sc.read_h5ad(CHECKPOINT_STEP04)
    step05_skip = False
else:
    # If Step 04 is also missing (prevent errors)
    step05_skip = False
    print("⚠ [Step 05] Warning: the Step 04 checkpoint was not found. Use adata currently in memory or check the preceding step first.")


# %%
# ==============================================================================
# [Step 05-2] Filter DLBCL data and construct KNN frequencies
# ==============================================================================
step05_ckpt_path = os.path.join(CKPT_DIR, 'mv1_step05_comp.pkl')

# 🌟 Restore required data even when skipping
if step05_skip:
    if os.path.exists(step05_ckpt_path):
        import joblib
        bkp = joblib.load(step05_ckpt_path)
        X_comp = bkp['X_comp']
        unique_labels = bkp['unique_labels']
        # Also restore cell_ids, which are needed for Sankey visualizations, etc. (request recalculation if missing).
        if 'cell_ids' in bkp:
            cell_ids = bkp['cell_ids']
        print(f"  ✅ Step 05-2 skipped: restored X_comp({X_comp.shape}) from disk.")
    else:
        if STEP05_FIGURE_ONLY:
            cn_cols = [c for c in adata.obs.columns if c.startswith('CN_K')]
            if not cn_cols:
                raise FileNotFoundError(
                    f"STEP05_FIGURE_ONLY=True but {step05_ckpt_path} is missing and no CN_K columns are available."
                )
            cell_ids = adata.obs.index[adata.obs[cn_cols[0]].notna()].tolist()
            unique_labels = sorted(adata.obs.get('Niche_Label', pd.Series(index=adata.obs.index, dtype=str)).dropna().astype(str).unique().tolist())
            X_comp = np.empty((len(cell_ids), max(len(unique_labels), 1)), dtype=np.float32)
            X_comp[:] = np.nan
            print("  ⚠️ No X_comp backup: skip CN/NMF quantitative recalculation and restore only spatial figures from checkpoint CN_K columns.")
        else:
            print("  ⚠️ Skip mode is active but the required backup (pkl) is missing. Forcing a rerun.")
            step05_skip = False

if not step05_skip:
    t_start = time.time()
    K_NEIGHBORS = 20
    K_SEARCH_MIN, K_SEARCH_MAX = 4, 12
    SUBTYPE_COL = 'ML_Subtype_Raw'

    # (Details omitted: data filtering and KNN frequency construction remain as before.)
    mask_dlbcl = adata.obs['Disease'].str.contains('DLBCL', na=False) & (adata.obs['Run_Tissue_name'] != 'DLBCL1_02')
    idx_dlbcl = np.where(mask_dlbcl)[0]
    niche_label = adata.obs[SUBTYPE_COL].astype(str).copy()
    if 'ML_Subtype_Raw' in adata.obs.columns:
        dlbcl_cells = adata.obs['Disease'].str.contains('DLBCL', na=False)
        niche_label[dlbcl_cells] = adata.obs.loc[dlbcl_cells, 'ML_Subtype_Raw'].astype(str)
    unknown_mask = niche_label.isin(['Unknown', 'na'])
    niche_label[unknown_mask] = adata.obs['Modeling_Broad'][unknown_mask].astype(str)
    adata.obs['Niche_Label'] = niche_label

    dlbcl_tissues = sorted(adata.obs.loc[mask_dlbcl, 'Run_Tissue_name'].unique())
    unique_labels = sorted(adata.obs.loc[mask_dlbcl, 'Niche_Label'].unique().tolist())
    unique_labels = [l for l in unique_labels if l not in ['Unknown', 'na']]
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    N_TYPES = len(unique_labels)

    freq_list, cell_ids = [], []
    for tissue in tqdm(dlbcl_tissues, desc="KNN Graph per Tissue"):
        t_mask = mask_dlbcl & (adata.obs['Run_Tissue_name'] == tissue)
        t_idx = np.where(t_mask)[0]
        coords = adata.obs.iloc[t_idx][['X_centroid', 'Y_centroid']].values.astype(np.float32)
        labels_arr = adata.obs['Niche_Label'].iloc[t_idx].map(label_to_idx)
        valid = ~labels_arr.isna()
        if valid.sum() < K_NEIGHBORS: continue
        labels_arr = labels_arr[valid].values.astype(np.int32)
        coords = coords[valid.values]; valid_global_idx = t_idx[valid.values]

        nn_model = cuNN(n_neighbors=K_NEIGHBORS)
        nn_model.fit(cp.array(coords))
        _, indices_gpu = nn_model.kneighbors(cp.array(coords))
        indices = cp.asnumpy(indices_gpu)
        neighbor_labels = labels_arr[indices]
        one_hot = np.eye(N_TYPES)[neighbor_labels]
        freqs = one_hot.sum(axis=1)
        freq_list.append(freqs); cell_ids.extend(adata.obs_names[valid_global_idx].tolist())

    X_freq = np.vstack(freq_list)
    row_sums = X_freq.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_comp = X_freq / row_sums

    # 🌟 Save for the next run (including cell_ids)
    import joblib
    joblib.dump({'X_comp': X_comp, 'unique_labels': unique_labels, 'cell_ids': cell_ids}, step05_ckpt_path)
    print(f"  ✅ [Backup complete] X_comp and metadata saved: {step05_ckpt_path}")

# %%
# ==============================================================================
# [Step 05-3] Search for the optimal K using multiple metrics
# ▷ Four metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz, and Inertia
# ▷ Automatically select the optimal K using Consensus Score = (Sil + DB(rev) + CH) / 3
# ▷ GPU-accelerated cuKMeans and sampled Silhouette computation
# ==============================================================================
from sklearn.metrics import silhouette_score

if 'X_comp' not in locals() and 'X_comp' not in globals():
    print("🔹 Loading from disk to restore X_comp...")
    step05_ckpt_path = os.path.join(CKPT_DIR, 'mv1_step05_comp.pkl')
    if os.path.exists(step05_ckpt_path):
        import joblib
        bkp5 = joblib.load(step05_ckpt_path)
        X_comp = bkp5['X_comp']
        unique_labels = bkp5['unique_labels']
        print(f"  ✅ Restoration complete (X_comp shape: {X_comp.shape})")
    else:
        raise ValueError("⚠️ The Step 05-2 checkpoint is missing! Run Step 05-2 at least once.")

if STEP05_FIGURE_ONLY:
    print("\n🔹 [Step 05-3] Figure-only mode → skip the KMeans optimal-K search and restore saved CN_K labels.")
    if 'K_SEARCH_MIN' not in globals() or 'K_SEARCH_MAX' not in globals():
        K_SEARCH_MIN, K_SEARCH_MAX = 4, 12
    K_SELECTED = [4, 6, 8, 10, 12]
    if 'cell_ids' not in globals():
        cn_seed_cols = [c for c in adata.obs.columns if c.startswith('CN_K') and not c.endswith('_smooth')]
        if not cn_seed_cols:
            raise KeyError("STEP05_FIGURE_ONLY=True but no CN_K columns are available to restore.")
        cell_ids = adata.obs.index[adata.obs[cn_seed_cols[0]].notna()].tolist()
    all_k_full_labels = {}
    for k in range(K_SEARCH_MIN, K_SEARCH_MAX + 1):
        col_k = f'CN_K{k}'
        if col_k in adata.obs.columns:
            all_k_full_labels[k] = adata.obs.loc[cell_ids, col_k].astype(str).tolist()
    missing_selected = [k for k in K_SELECTED if k not in all_k_full_labels]
    if missing_selected:
        raise KeyError(f"STEP05_FIGURE_ONLY=True but required CN_K columns are missing: {missing_selected}")
    best_k = max(K_SELECTED)
    k_range = range(K_SEARCH_MIN, K_SEARCH_MAX + 1)
    metrics = {'Inertia': [], 'Silhouette': [], 'Davies_Bouldin': [], 'Calinski_Harabasz': []}
else:
    print("\n🔹 [Step 05-3] Searching for the optimal K using multiple metrics (GPU-accelerated, all DLBCL cells)...")
    metrics = {'Inertia': [], 'Silhouette': [], 'Davies_Bouldin': [], 'Calinski_Harabasz': []}

    # Record all labels for Sankey mapping
    all_k_full_labels = {} 

    # Define direct GPU (CuPy) metric computation functions (memory-optimized and fast)
    def calinski_harabasz_cupy(X_cp, labels_cp, k_model):
        n_samples, _ = X_cp.shape
        n_clusters = k_model.n_clusters
    
        global_mean = cp.mean(X_cp, axis=0) # Global mean
        centers = k_model.cluster_centers_  # Cluster centers
    
        _, counts = cp.unique(labels_cp, return_counts=True)
        # Between-group scatter (BGS)
        bgs = cp.sum(counts[:, None] * ((centers - global_mean) ** 2))
        # Within-group scatter (WGS: use Inertia)
        wgs = k_model.inertia_
    
        if wgs == 0: return 1.0
        return float((bgs / (n_clusters - 1)) / (wgs / (n_samples - n_clusters)))

    def davies_bouldin_cupy(X_cp, labels_cp, k_model):
        n_clusters = k_model.n_clusters
        centers = k_model.cluster_centers_
    
        # Calculate the mean distance between each data point and its cluster center
        S = cp.zeros(n_clusters)
        for i in range(n_clusters):
            mask = (labels_cp == i)
            if cp.any(mask):
                diff = X_cp[mask] - centers[i]
                S[i] = cp.mean(cp.sqrt(cp.sum(diff**2, axis=1)))
            
        db_score = 0.0
        for i in range(n_clusters):
            max_ratio = 0.0
            for j in range(n_clusters):
                if i != j:
                    M_ij = cp.sqrt(cp.sum((centers[i] - centers[j])**2))
                    if M_ij > 0:
                        ratio = (S[i] + S[j]) / M_ij
                        if ratio > max_ratio:
                            max_ratio = ratio
            db_score += max_ratio
        return float(db_score / n_clusters)


    cp.random.seed(42)

    if 'K_SEARCH_MIN' not in globals() or 'K_SEARCH_MAX' not in globals():
        K_SEARCH_MIN, K_SEARCH_MAX = 4, 12

    k_range = range(K_SEARCH_MIN, K_SEARCH_MAX + 1)
    X_cp = cp.array(X_comp) # Load the data onto the GPU

    for k in tqdm(k_range, desc="GPU Optimization Loop"):
        k_model = cuKMeans(n_clusters=k, max_iter=300, random_state=42)
        k_model.fit(X_cp)
    
        # Predicted labels for all DLBCL cells (GPU memory)
        full_lbls_cp = k_model.predict(X_cp)
    
        # For Sankey mapping (convert to CPU numpy and save)
        full_lbls_np = cp.asnumpy(full_lbls_cp)
        all_k_full_labels[k] = [f'K{k}_CN{l+1}' for l in full_lbls_np]
    
        # 🌟 Compute metrics rapidly on the GPU using all original data (X_comp)
        metrics['Inertia'].append(k_model.inertia_)
        metrics['Davies_Bouldin'].append(davies_bouldin_cupy(X_cp, full_lbls_cp, k_model))
        metrics['Calinski_Harabasz'].append(calinski_harabasz_cupy(X_cp, full_lbls_cp, k_model))
    
        # Silhouette Score (sample 50,000 representative cells due to O(N²) computational cost)
        if X_cp.shape[0] > 50000:
            idx = cp.random.choice(X_cp.shape[0], 50000, replace=False)
            sil_score = silhouette_score(cp.asnumpy(X_cp[idx]), cp.asnumpy(full_lbls_cp[idx]))
        else:
            sil_score = silhouette_score(cp.asnumpy(X_cp), cp.asnumpy(full_lbls_cp))
        metrics['Silhouette'].append(sil_score)


    # Normalize and sum
    def normalize(data, reverse=False):
        arr = np.array(data); norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return 1 - norm if reverse else norm

    consensus_score = (normalize(metrics['Silhouette']) + 
                       normalize(metrics['Davies_Bouldin'], reverse=True) + 
                       normalize(metrics['Calinski_Harabasz'])) / 3
    best_k = list(k_range)[np.argmax(consensus_score)]

    print(f"\n ⭐ Consensus Optimal K: {best_k}")

    # 1. Plot individual metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='white')
    axes[0,0].plot(k_range, metrics['Inertia'], 'go-'); axes[0,0].set_title('Inertia (Elbow)')
    axes[0,1].plot(k_range, metrics['Silhouette'], 'bo-'); axes[0,1].set_title('Silhouette (Higher is Better)')
    axes[0,1].axvline(best_k, color='red', ls='--')
    axes[1,0].plot(k_range, metrics['Davies_Bouldin'], 'yo-'); axes[1,0].set_title('Davies-Bouldin (Lower is Better)')
    axes[1,0].axvline(best_k, color='red', ls='--')
    axes[1,1].plot(k_range, metrics['Calinski_Harabasz'], 'mo-'); axes[1,1].set_title('Calinski-Harabasz (Higher is Better)')
    axes[1,1].axvline(best_k, color='red', ls='--')
    plt.tight_layout()
    # Excluded from the Final Figure list; disable only display.
    # plt.show()

    # 2. Combined Consensus Score plot
    plt.figure(figsize=(10, 6), facecolor='white')
    plt.plot(k_range, consensus_score, 'k^-', markersize=10, linewidth=2)
    plt.axvline(best_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K = {best_k}')
    plt.title('Normalized Consensus Score (Silhouette + DB(rev) + CH)', fontsize=14)
    plt.xlabel('Number of Clusters (K)'); plt.ylabel('Consensus Score (0~1)')
    plt.legend(); plt.grid(True, alpha=0.3)
    # Excluded from the Final Figure list; disable only display.
    # plt.show()



# %%
# ==============================================================================
# [Step 05-4] CN Evolution & Lineage Analysis (Sankey visualization and adata.obs mapping)
# ▷ Calculate cluster transition matrices across K = [4, 6, 8, 10, 12]
# ▷ Organize neighborhood composition labels (Top 3) and plot a Sankey diagram
# ==============================================================================

import plotly.graph_objects as go
import plotly.express as px
import IPython
import numpy as np
import pandas as pd

K_SELECTED = [4, 6, 8, 10, 12] # K values to track
print(f"🔹 Visualizing CN evolution and neighborhood composition / mapping to adata...")

# 1. Calculate neighborhood composition labels (Top 3) and save mappings to adata.obs
cn_info = {}

# Load lineage mapping data obtained for all DLBCL cells (X_comp) in Step 05-3
flow_df = pd.DataFrame(all_k_full_labels) 

# Convert X_comp to a DataFrame once for fast Top3 calculations
tmp_df = pd.DataFrame(X_comp, columns=unique_labels)

for k_idx, k in enumerate(K_SELECTED):
    print(f"  ▶ Mapping DLBCL CN assignments and analyzing neighborhood composition for K = {k}...")
    
    # 🌟 Insert precisely at the cell_ids positions filtered from the full 5.7-million-cell AnnData
    adata.obs.loc[cell_ids, f'CN_K{k}'] = all_k_full_labels[k]
    
    # Use the labels already generated to avoid duplicate K-Means predictions
    curr_labels_arr = np.array(all_k_full_labels[k])
    
    for l in range(k):
        node_id = f"K{k}_CN{l+1}"
        cn_info[node_id] = {"top3": "", "origins": ""}
        
        # Extract cells belonging to the current cluster
        c_mask = (curr_labels_arr == node_id)
        
        if c_mask.any():
            top3 = tmp_df[c_mask].mean().sort_values(ascending=False).head(3)
            # Combine for clean text rendering
            cn_info[node_id]["top3"] = " | ".join([f"{idx}({val*100:.1f}%)" for idx, val in top3.items()])

    # [Origins] Calculate transition proportions (Origins) from the previous K value
    if k_idx > 0:
        prev_k = K_SELECTED[k_idx - 1]
        lineage_stats = flow_df.groupby([prev_k, k]).size().unstack(fill_value=0)
        # Convert to percentages (pct)
        lineage_pct = (lineage_stats.divide(lineage_stats.sum(axis=0), axis=1) * 100).round(1)
        
        for child_node in lineage_pct.columns:
            if child_node in cn_info:
                # Show only origins exceeding 1% in the text
                origins = lineage_pct[child_node].sort_values(ascending=False)
                origins = origins[origins > 1]
                origin_str = " + ".join([f"K{prev_k}_{n.split('_')[1]}({v}%)" for n, v in origins.items()])
                cn_info[child_node]["origins"] = origin_str

print("✅ Lineage calculations and saving mappings to adata.obs are complete.")

# 2. Map Sankey nodes and set colors
node_list = []
for k in K_SELECTED: 
    node_list.extend(flow_df[k].unique())
# Sort nodes by K value, then CN value
node_list = sorted(list(set(node_list)), key=lambda x: (int(x.split('_')[0][1:]), int(x.split('_')[1][2:])))
node_map = {node: i for i, node in enumerate(node_list)}

color_palette = px.colors.qualitative.Pastel + px.colors.qualitative.Safe
# Assign colors to the initial nodes (K=4)
color_mapping = {n: color_palette[i % len(color_palette)] for i, n in enumerate(flow_df[K_SELECTED[0]].unique())}

# 3. Link data and parent-based color inheritance
s_sources, s_targets, s_values, s_link_colors = [], [], [], []

for i in range(len(K_SELECTED) - 1):
    c_k, n_k = K_SELECTED[i], K_SELECTED[i+1]
    trans_matrix = flow_df.groupby([c_k, n_k]).size().unstack(fill_value=0)
    
    # Assign the main parent node's color to each child node by majority rule
    for child in trans_matrix.columns:
        major_parent = trans_matrix[child].idxmax()
        if child not in color_mapping:
            color_mapping[child] = color_mapping.get(major_parent, "gray")

    # Record link flows (data transitions)
    trans_data = flow_df.groupby([c_k, n_k]).size().reset_index(name='val')
    for _, r in trans_data.iterrows():
        src, tar = r[c_k], r[n_k]
        if r['val'] > 0:
            s_sources.append(node_map[src])
            s_targets.append(node_map[tar])
            s_values.append(r['val'])
            # Use transparent link colors (rgba) reflecting the parent node color
            p_color = color_mapping.get(src, "rgba(200, 200, 200, 0.4)")
            if 'rgb' in p_color:
                s_link_colors.append(p_color.replace('rgb', 'rgba').replace(')', ', 0.35)'))
            else:
                s_link_colors.append(p_color)

# Format node text (using HTML Span tags)
node_labels = [
    f"<b>{n}</b><br>"
    f"<span style='font-size:10px; color:gray;'>"
    f"Prev: {cn_info[n]['origins'] if cn_info[n]['origins'] else 'Root'}<br>"
    f"Top composition: {cn_info[n]['top3']}</span>" 
    for n in node_list
]

# 4. Output the Sankey diagram
fig_sankey = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=20, 
        thickness=30,
        label=node_labels,
        color=[color_mapping.get(n, "gray") for n in node_list],
        line=dict(color="black", width=0.5)
    ),
    link=dict(
        source=s_sources, 
        target=s_targets, 
        value=s_values, 
        color=s_link_colors
    )
))

fig_sankey.update_layout(
    title="<b>Detailed CN Evolution & Neighborhood Composition Flow</b>", 
    font_size=11, 
    width=1500, 
    height=900,
    margin=dict(t=50, b=50, l=150, r=150)
)

# Step05 figures are excluded from the Final Figure list; disable only notebook display.
# IPython.display.display(IPython.display.HTML(fig_sankey.to_html()))


# %%
# ==============================================================================
# [Step 05-5] Separate spatial visualizations for individual DLBCL slices
# ▷ Plot each tissue as a scatter plot with colors indicating CN labels
# ▷ Use the Sankey palette for a consistent color scheme
# ==============================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# 🌟 Important: change this to the column that identifies samples (slices) in the actual data!
sample_col = 'Run_Tissue_name'  # Examples: 'FOV', 'Image', 'Core', 'Sample', 'Point', etc.

# 1. Check that the column exists
if sample_col not in adata.obs.columns:
    print(f"⚠️ Column '{sample_col}' is missing from adata.obs. Update it if the actual name differs.")
    print(f"Available columns: {adata.obs.columns.tolist()[:15]}...")
else:
    # 2. Use the Sankey palette
    def to_mpl_color(color_str):
        if color_str.startswith('rgb'):
            clean_str = color_str.replace('rgba', '').replace('rgb', '').replace('(', '').replace(')', '')
            vals = [int(v.strip()) for v in clean_str.split(',')[:3]] 
            return "#{:02x}{:02x}{:02x}".format(vals[0], vals[1], vals[2])
        return color_str
    mpl_color_map = {node: to_mpl_color(color) for node, color in color_mapping.items()}

    # 3. DLBCL data and target resolution (K=12, the most detailed map, as an example here)
    dlbcl_obs = adata.obs.loc[cell_ids].copy()
    k_to_plot = 12
    col_k = f'CN_K{k_to_plot}'
    
    # 💡 [Added] Specify coordinate variables
    x_col = 'X_centroid'  
    y_col = 'Y_centroid'
    
    # 4. Allocate a separate subplot canvas for each sample
    unique_samples = dlbcl_obs[sample_col].unique()
    num_samples = len(unique_samples)
    cols = 5 # Five tissues per row
    rows = math.ceil(num_samples / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=300)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    print(f"Drawing separate images for {num_samples} DLBCL slices (tissues) in total...")
    
    for i, seq_sample in enumerate(unique_samples):
        ax = axes[i]
        sample_mask = (dlbcl_obs[sample_col] == seq_sample)
        sample_obs = dlbcl_obs[sample_mask]
        
        for cn_label in sorted(sample_obs[col_k].unique()):
            cn_mask = (sample_obs[col_k] == cn_label)
            node_color = mpl_color_map.get(cn_label, 'gray')
            
            ax.scatter(sample_obs.loc[cn_mask, x_col], 
                       sample_obs.loc[cn_mask, y_col], 
                       c=node_color, s=2.0, alpha=0.9, edgecolors='none')
            
        ax.set_title(f"Sample: {seq_sample}", fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        
    # Remove unused blank canvases
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    # Excluded from the Final Figure list; disable only display.
    # plt.show()


# %%
# ==============================================================================
# [Step 05-6] Continuous Map & Dual-Control (Unique & Consistent Color Fix)
# ▷ Rendering pipeline guaranteeing 100% unique colors
# ▷ Generate a 3-panel HTML view: Sankey + spatial map + DotPlot
# ==============================================================================

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy import stats
from scipy.spatial import cKDTree
import math
import os
import IPython

print("🔹 [Step 05-6] Running the rendering pipeline guaranteeing 100% unique colors...")

sample_col = 'Run_Tissue_name'  
K_SELECTED = [4, 6, 8, 10, 12] 

# =====================================================================
# 📌 PART 1: Sankey diagram and dominant CN composition mapping
# =====================================================================
cn_info, flow_df = {}, pd.DataFrame(all_k_full_labels) 
tmp_df = pd.DataFrame(X_comp, columns=unique_labels)

# 1. Calculate text labels and statistics
for k_idx, k in enumerate(K_SELECTED):
    adata.obs.loc[cell_ids, f'CN_K{k}'] = all_k_full_labels[k]
    curr_labels_arr = np.array(all_k_full_labels[k])
    for l in range(k):
        node_id = f"K{k}_CN{l+1}"
        cn_info[node_id] = {"top3": "", "origins": ""}
        c_mask = (curr_labels_arr == node_id)
        if c_mask.any():
            top3 = tmp_df[c_mask].mean().sort_values(ascending=False).head(3)
            cn_info[node_id]["top3"] = " | ".join([f"{idx}({val*100:.1f}%)" for idx, val in top3.items()])

    if k_idx > 0:
        prev_k = K_SELECTED[k_idx - 1]
        lineage_stats = flow_df.groupby([prev_k, k]).size().unstack(fill_value=0)
        lineage_pct = (lineage_stats.divide(lineage_stats.sum(axis=0), axis=1) * 100).round(1)
        for child_node in lineage_pct.columns:
            if child_node in cn_info:
                origins = lineage_pct[child_node].sort_values(ascending=False)
                origins = origins[origins > 1]
                cn_info[child_node]["origins"] = " + ".join([f"K{prev_k}_{n.split('_')[1]}({v}%)" for n, v in origins.items()])

node_list = []
for k in K_SELECTED: node_list.extend(flow_df[k].unique())
node_list = sorted(list(set(node_list)), key=lambda x: (int(x.split('_')[0][1:]), int(x.split('_')[1][2:])))
node_map = {node: i for i, node in enumerate(node_list)}

# 🌟 2. Dynamic unique-color assignment to prevent overlap (Dominant & Branch)
# 🌟 Unique-color assignment to completely prevent overlap (Dominant & Branch)
# Combine over 45 refined spectrum colors to ensure no repeated colors within a region
master_palette = (
    px.colors.qualitative.Set1 +     # Nine classic primary colors with high clarity
    px.colors.qualitative.Vivid +    # Eleven vivid, bright marker-fluorescence colors
    px.colors.qualitative.Bold +     # Ten visually strong solid colors
    px.colors.qualitative.Plotly +   # Ten basic, intense primary colors
    px.colors.qualitative.Dark24     # Twenty-four dark tones useful for distinguishing boundaries
)
master_palette = list(dict.fromkeys(master_palette)) # Completely remove duplicate colors
color_mapping = {}
available_colors = master_palette.copy()
# 1. Assign colors to the initial clusters (K=4)
for n in flow_df[K_SELECTED[0]].unique():
    color_mapping[n] = available_colors.pop(0)
# 2. Calculate colors for clusters derived during splitting
for i in range(len(K_SELECTED) - 1):
    c_k, n_k = K_SELECTED[i], K_SELECTED[i+1]
    trans_matrix = flow_df.groupby([c_k, n_k]).size().unstack(fill_value=0)
    
    # Track the main parent that contributed the largest share (cells) to each child
    child_to_parent = {child: trans_matrix[child].idxmax() for child in trans_matrix.columns}
    
    # Group the derived children by parent
    parent_to_children = {}
    for c, p in child_to_parent.items():
        parent_to_children.setdefault(p, []).append(c)
        
    for p, children in parent_to_children.items():
        # Sort descending by the number of cells inherited from the parent (main-line order)
        children_sorted = sorted(children, key=lambda x: trans_matrix.loc[p, x], reverse=True)
        
        # 👑 The first, main child (Dominant) retains the original parent color
        color_mapping[children_sorted[0]] = color_mapping[p]
        
        # 🌿 Assign new, independent, non-overlapping colors to all other children (New Branch)
        for child in children_sorted[1:]:
            color_mapping[child] = available_colors.pop(0)

s_sources, s_targets, s_values, s_link_colors = [], [], [], []
for i in range(len(K_SELECTED) - 1):
    c_k, n_k = K_SELECTED[i], K_SELECTED[i+1]
    trans_data = flow_df.groupby([c_k, n_k]).size().reset_index(name='val')
    for _, r in trans_data.iterrows():
        src, tar = r[c_k], r[n_k]
        if r['val'] > 0:
            s_sources.append(node_map[src]); s_targets.append(node_map[tar]); s_values.append(r['val'])
            p_color = color_mapping.get(src, "rgba(200, 200, 200, 0.4)")
            s_link_colors.append(p_color.replace('rgb', 'rgba').replace(')', ', 0.35)') if 'rgb' in p_color else p_color)

node_labels = [f"<b>{n}</b><br><span style='font-size:10px; color:gray;'>Prev: {cn_info[n]['origins'] if cn_info[n]['origins'] else 'Root'}<br>Top composition: {cn_info[n]['top3']}</span>" for n in node_list]

fig_sankey = go.Figure(go.Sankey(arrangement="snap", node=dict(pad=20, thickness=30, label=node_labels, color=[color_mapping.get(n, "gray") for n in node_list], line=dict(color="black", width=0.5)), link=dict(source=s_sources, target=s_targets, value=s_values, color=s_link_colors)))
fig_sankey.update_layout(title="<b>[1] Detailed CN Evolution Matrix (Neighborhood Composition Clusters)</b>", font_size=11, width=1500, height=800, margin=dict(t=50, b=30, l=150, r=150))

# =====================================================================
# 📌 PART 2: Spatial coordinates (continuous-map pixel interpolation rendering)
# 💡 [Key change] Build the KD-Tree from dlbcl_obs (DLBCL only), replacing all_obs (mixed references)
#    Map every pixel only to cells with CN labels to generate continuous CN maps
# =====================================================================
x_col = next((col for col in ['X_centroid','CenterX','X','x','x_centroid'] if col in adata.obs.columns), None)
y_col = next((col for col in ['Y_centroid','CenterY','Y','y','y_centroid'] if col in adata.obs.columns), None)

def hex_to_rgba(h, a=255):
    h = h.replace('rgba', '').replace('rgb', '').replace('(', '').replace(')', '').strip()
    if ',' in h: 
        vals = [int(v.strip()) for v in h.split(',')]
        return np.array([vals[0], vals[1], vals[2], a], dtype=np.uint8)
    h = h.lstrip('#')
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a], dtype=np.uint8)

color_rgba_map = {n: hex_to_rgba(c) for n, c in color_mapping.items()}
mpl_color_map = {n: f"rgb({c[0]},{c[1]},{c[2]})" for n, c in color_rgba_map.items()}
bg_rgba = np.array([220, 220, 220, 150], dtype=np.uint8)

dlbcl_obs = adata.obs.loc[cell_ids].copy()
unique_samples = dlbcl_obs[sample_col].unique()

# 💡 [Preserve the original tissue shape] Retrieve all-cell information again for masking
all_obs = adata.obs[adata.obs[sample_col].isin(unique_samples)].copy()

# 💡 [Change] Region smoothing (increase K=15 → K=30 for smoother continuous maps)
SMOOTH_K = 30
for s in unique_samples:
    mask = dlbcl_obs[sample_col] == s
    s_coords = dlbcl_obs.loc[mask, [x_col, y_col]].values
    if len(s_coords) < SMOOTH_K: 
        for k in K_SELECTED: dlbcl_obs.loc[mask, f'CN_K{k}_smooth'] = dlbcl_obs.loc[mask, f'CN_K{k}']
        dlbcl_obs.loc[mask, 'Niche_Label_smooth'] = dlbcl_obs.loc[mask, 'Niche_Label']
        continue
    tree = cKDTree(s_coords)
    _, indices = tree.query(s_coords, k=SMOOTH_K) 
    for k in K_SELECTED:
        cn_labels = dlbcl_obs.loc[mask, f'CN_K{k}'].to_numpy()
        u_labels, codes = np.unique(cn_labels, return_inverse=True)
        mode_res = stats.mode(codes[indices], axis=1, keepdims=False)
        dlbcl_obs.loc[mask, f'CN_K{k}_smooth'] = u_labels[mode_res.mode.flatten() if hasattr(mode_res, 'mode') else mode_res[0].flatten()]
    
    # [Added] Subtype (Niche_Label) smoothing
    sub_labels = dlbcl_obs.loc[mask, 'Niche_Label'].astype(str).to_numpy()
    u_labels_sub, codes_sub = np.unique(sub_labels, return_inverse=True)
    mode_res_sub = stats.mode(codes_sub[indices], axis=1, keepdims=False)
    dlbcl_obs.loc[mask, 'Niche_Label_smooth'] = u_labels_sub[mode_res_sub.mode.flatten() if hasattr(mode_res_sub, 'mode') else mode_res_sub[0].flatten()]

# Calculate coordinate bounds using the full tissue (all_obs) to include the entire tissue shape
g_min_x, g_max_x = all_obs[x_col].min(), all_obs[x_col].max()
g_min_y, g_max_y = all_obs[y_col].min(), all_obs[y_col].max()
max_w, max_h = g_max_x - g_min_x, g_max_y - g_min_y

cols = 4; rows = math.ceil(len(unique_samples) / cols)
pad_x, pad_y = max_w * 0.15, max_h * 0.25
all_obs['X_plot'], all_obs['Y_plot'] = 0.0, 0.0
annotations = []

for i, s in enumerate(unique_samples):
    r, c = int(i / cols), i % cols
    mask = all_obs[sample_col] == s
    offset_x, offset_y = c * (max_w + pad_x), (rows - 1 - r) * (max_h + pad_y) 
    all_obs.loc[mask, 'X_plot'] = (all_obs.loc[mask, x_col] - g_min_x) + offset_x
    all_obs.loc[mask, 'Y_plot'] = (g_max_y - all_obs.loc[mask, y_col]) + offset_y 
    annotations.append(dict(x=offset_x + max_w/2, y=offset_y + max_h + pad_y/3, xref="x", yref="y", text=f"<b>{s}</b>", showarrow=False, font=dict(size=14), xanchor='center'))

# Synchronize the calculated positions to DLBCL_obs (preserve index alignment)
dlbcl_obs['X_plot'] = all_obs['X_plot']
dlbcl_obs['Y_plot'] = all_obs['Y_plot']

margin_l, margin_r, margin_t, margin_b = 50, 50, 100, 50
p_xmin, p_xmax = all_obs['X_plot'].min() - (max_w * 0.1), all_obs['X_plot'].max() + (max_w * 0.1)
p_ymin, p_ymax = all_obs['Y_plot'].min() - (max_h * 0.1), all_obs['Y_plot'].max() + (max_h * 0.1)

c_width, c_height = p_xmax - p_xmin, p_ymax - p_ymin
set_width = 1600; target_w = 2200 
px_scale = target_w / c_width
target_h = int(c_height * px_scale)
set_height = int(c_height * ((set_width - margin_l - margin_r) / c_width)) + margin_t + margin_b

xs, ys = np.linspace(p_xmin, p_xmax, target_w), np.linspace(p_ymax, p_ymin, target_h)
xv, yv = np.meshgrid(xs, ys)
query_pts = np.c_[xv.ravel(), yv.ravel()]

# 🔥 1. Shape masking and full subtype colors (based on all cells)
all_obs_tree = cKDTree(all_obs[['X_plot', 'Y_plot']].values)
mask_dists, nearest_idx_all = all_obs_tree.query(query_pts)
inter_cell_dist, _ = all_obs_tree.query(all_obs[['X_plot', 'Y_plot']].values, k=2)
# Apply the 99th percentile using densely spaced all-cell distances to produce clean edges
cutoff = np.percentile(inter_cell_dist[:, 1], 99) * 2.5 
is_valid = mask_dists < cutoff
valid_indices_all = nearest_idx_all[is_valid]

# 🔥 2. Nearest cells used for CN colors (based on DLBCL)
cell_tree = cKDTree(dlbcl_obs[['X_plot', 'Y_plot']].values)
_, nearest_idx = cell_tree.query(query_pts)
valid_indices = nearest_idx[is_valid]

traces, trace_k_mapping = [], []
map_formats = [f'CN_K{k}_smooth' for k in K_SELECTED] + ['Subtype']

for i, col_name in enumerate(map_formats):
    if 'CN_K' in col_name:
        labels_arr = dlbcl_obs[col_name].astype(str).values
        k_val = col_name.split('_')[1][1:]
        trace_name = f"K={k_val} Map"
        legend_base = trace_name
        color_lookup = {lbl: color_rgba_map.get(lbl, bg_rgba) for lbl in np.unique(labels_arr)}
        scatter_marker_dict = lambda lbl: dict(color=mpl_color_map.get(lbl, 'gray'), size=15, symbol='square')
        trace_tag = int(k_val)
        
        # [Change] Show individual cells clearly as dots rather than blending mosaic pixels
        img_1d = np.zeros((target_h * target_w, 4), dtype=np.uint8)
        img_1d[is_valid] = [0, 0, 0, 255]  # Use a black (or transparent) background
        
        px_x = np.clip(np.round((dlbcl_obs['X_plot'].values - p_xmin) / c_width * (target_w - 1)).astype(int), 0, target_w - 1)
        px_y = np.clip(np.round((p_ymax - dlbcl_obs['Y_plot'].values) / c_height * (target_h - 1)).astype(int), 0, target_h - 1)
        
        for lbl in np.unique(labels_arr):
            l_mask = (labels_arr == lbl)
            clr = color_lookup[lbl]
            xs, ys = px_x[l_mask], px_y[l_mask]
            sz = 2  # Dot size
            for dy in range(sz):
                for dx in range(sz):
                    nx = np.clip(xs + dx, 0, target_w - 1)
                    ny = np.clip(ys + dy, 0, target_h - 1)
                    img_1d[ny * target_w + nx] = clr
                    
        unique_leg_labels = sorted(np.unique(labels_arr))
        
    else:  # Subtype Map
        # Plot subtypes directly as dense dots (scatter rasterization) to avoid immune-cell overlap and improve visibility
        labels_arr = all_obs['Niche_Label'].astype(str).values
        trace_name = "Subtype Map"
        legend_base = "Subtype"
        
        # Base color mapping
        color_lookup = {lbl: hex_to_rgba(get_color(lbl)) for lbl in np.unique(labels_arr)}
        # Improve visibility by placing B cells (the main tumor component) in the background as pale pastel transparent blocks (alpha 150)
        bg_bcell_rgba = np.array([235, 245, 251, 150], dtype=np.uint8) 
        if 'B cell' in color_lookup: color_lookup['B cell'] = bg_bcell_rgba
        if 'B cells' in color_lookup: color_lookup['B cells'] = bg_bcell_rgba
        
        def sub_color(lbl):
            if lbl in ['B cell', 'B cells']: return '#EBF5FB'
            return get_color(lbl)
            
        scatter_marker_dict = lambda lbl: dict(color=sub_color(lbl), size=15, symbol='square')
        trace_tag = 'Subtype'
        
        # Initialize the background as black only inside the tissue mask (is_valid), leaving the rest transparent
        img_1d = np.zeros((target_h * target_w, 4), dtype=np.uint8)
        img_1d[is_valid] = [0, 0, 0, 255]
        
        # Calculate pixel coordinates (reverse Y because, as in query_pts, it starts at p_ymax)
        px_x = np.clip(np.round((all_obs['X_plot'].values - p_xmin) / c_width * (target_w - 1)).astype(int), 0, target_w - 1)
        px_y = np.clip(np.round((p_ymax - all_obs['Y_plot'].values) / c_height * (target_h - 1)).astype(int), 0, target_h - 1)
        
        # Adjust rendering order: draw B-cell lineages first as the base, then immune cells on top for emphasis
        draw_order = sorted(np.unique(labels_arr), key=lambda x: 0 if 'B cell' in x else 1)
        
        for lbl in draw_order:
            l_mask = (labels_arr == lbl)
            clr = color_lookup[lbl]
            xs, ys = px_x[l_mask], px_y[l_mask]
            
            # Assign sizes: detailed 1x1 dots for background B cells and more visible 2x2 dots for immune cells, etc.
            sz = 1 if 'B cell' in lbl else 2
            for dy in range(sz):
                for dx in range(sz):
                    nx = np.clip(xs + dx, 0, target_w - 1)
                    ny = np.clip(ys + dy, 0, target_h - 1)
                    img_1d[ny * target_w + nx] = clr
        
        # Because subtypes are numerous, include only representative labels for major lineages in the legend (Top 15 by abundance)
        counts = pd.Series(labels_arr).value_counts()
        unique_leg_labels = counts.head(15).index.tolist()
    
    # Create the image after flipping vertically
    img_2d = np.flipud(img_1d.reshape((target_h, target_w, 4)))
    traces.append(go.Image(
        z=img_2d, x0=p_xmin, dx=c_width/target_w, y0=p_ymin, dy=c_height/target_h,
        name=trace_name, visible=(i == 0)
    ))
    trace_k_mapping.append(trace_tag)
    
    for lbl in unique_leg_labels:
        disp_name = lbl.split('_')[1] if 'CN_K' in col_name else lbl
        if disp_name == 'B cell':
            disp_name = 'B cells'
        traces.append(go.Scatter(
            x=[None], y=[None], mode='markers', marker=scatter_marker_dict(lbl),
            name=disp_name, legendgroup=legend_base, visible=(i == 0)
        ))
        trace_k_mapping.append(trace_tag)

btn_k, btn_image = [], [dict(label="🔍 View All Regions", method="relayout", args=[{"xaxis.range": [p_xmin, p_xmax], "yaxis.range": [p_ymin, p_ymax]}])]
for tag in sorted(list(set([str(t) for t in trace_k_mapping]))):
    # Distinguish K values or Subtype when sorting
    visibility = [bool(str(t_k) == tag) for t_k in trace_k_mapping]
    btn_label = f"📍 Subtype Spatial Map" if tag == 'Subtype' else f"📍 CN Resolution (K = {tag})"
    btn_k.append(dict(label=btn_label, method="restyle", args=[{"visible": visibility}]))

for s in unique_samples:
    c_idx = list(unique_samples).index(s)
    r, c = int(c_idx / cols), c_idx % cols
    ox, oy = c * (max_w + pad_x), (rows - 1 - r) * (max_h + pad_y) 
    btn_image.append(dict(label=f"🔎 Zoom: {s}", method="relayout", args=[{"xaxis.range": [ox - max_w*0.05, ox + max_w*1.05], "yaxis.range": [oy - max_h*0.05, oy + max_h*1.05]}]))

fig_spatial = go.Figure(data=traces)
fig_spatial.update_layout(
    updatemenus=[
        dict(active=0, buttons=btn_k, direction="down", showactive=True, x=0.01, xanchor="left", y=1.05, yanchor="top", font=dict(size=14)),
        dict(active=0, buttons=btn_image, direction="down", showactive=True, x=0.25, xanchor="left", y=1.05, yanchor="top", font=dict(size=14))
    ],
    annotations=annotations, title=f"<b>[2] Interactive Spatial Map</b> (CN clusters and subtype-label reference map)", title_font_size=20, plot_bgcolor='white', hovermode='closest',
    width=set_width, height=set_height, margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b)
)
fig_spatial.update_xaxes(showgrid=False, zeroline=False, visible=False, range=[p_xmin, p_xmax])
fig_spatial.update_yaxes(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1, range=[p_ymin, p_ymax]) 

# =====================================================================
# 📌 PARTS 3 and 4: Dual-control dynamic marker Dotplot and HTML file generation
# =====================================================================
if sp.issparse(adata.X): expr_df = pd.DataFrame(adata[cell_ids].X.toarray(), index=cell_ids, columns=adata.var_names)
else: expr_df = pd.DataFrame(adata[cell_ids].X, index=cell_ids, columns=adata.var_names)

markers = list(adata.var_names)
dot_traces = []

for k in K_SELECTED:
    col_k_smooth = f'CN_K{k}_smooth'
    cns = sorted(dlbcl_obs[col_k_smooth].unique(), key=lambda x: int(x.split('_')[1][2:]))
    for c_type in unique_labels:
        type_mask = (dlbcl_obs['Niche_Label'] == c_type)
        c_expr = expr_df[type_mask]
        
        x_vals, y_vals, colors, sizes, hover_info = [], [], [], [], []
        for cn in cns[::-1]: 
            cn_mask = dlbcl_obs.loc[type_mask, col_k_smooth] == cn
            if cn_mask.sum() > 0:
                cn_expr_s = c_expr[cn_mask]
                mean_v, frac_v = cn_expr_s.mean(axis=0), (cn_expr_s > 0.0).mean(axis=0)
            else:
                mean_v, frac_v = pd.Series(0, index=markers), pd.Series(0, index=markers)
                
            for m in markers:
                x_vals.append(m); y_vals.append(cn.split('_')[1])
                colors.append(mean_v[m]); sizes.append(frac_v[m] * 120)  
                hover_info.append(f"<b>{c_type}</b><br>CN: {cn}<br>Marker: {m}<br>Mean Expr: {mean_v[m]:.2f}<br>Fraction: {frac_v[m]*100:.1f}%")
                
        is_val = bool((k == K_SELECTED[0]) and (c_type == unique_labels[0]))
        dot_traces.append(go.Scatter(
            x=x_vals, y=y_vals, mode='markers',
            marker=dict(size=sizes, sizemode='area', color=colors, colorscale='Reds', showscale=is_val, colorbar=dict(title="Mean", x=1.05) if is_val else None, line=dict(color='gray', width=0.5)),
            name=f"K={k}_{c_type}", visible=is_val, text=hover_info, hoverinfo="text"
        ))

fig_dotplot = go.Figure(data=dot_traces)
fig_dotplot.update_layout(xaxis=dict(tickangle=45, showgrid=True), yaxis=dict(showgrid=True, automargin=True), plot_bgcolor='white', width=1500, height=350 + (max(K_SELECTED)*18), margin=dict(t=20, b=50))
k_opts = "".join([f'<option value="{k}">Resolution K = {k}</option>' for k in K_SELECTED])
ctype_opts = "".join([f'<option value="{ct}">{ct}</option>' for ct in unique_labels])
html_dot_controls = f"""
<div style="font-family: Arial; width: 1500px; text-align: left; margin-bottom: -10px; margin-top: 20px;">
    <h3 style="display:inline-block; margin-right: 30px;">[3] Cell Type & CN Specific Marker Expression</h3>
    <div style="display:inline-block; background:#f0f0f0; padding:10px 15px; border-radius:5px; border:1px solid #ccc;">
        <label style="font-weight:bold;">📍 Select K:</label><select id="k_select" style="font-size:14px; margin-right:20px;">{k_opts}</select>
        <label style="font-weight:bold;">🦠 Select Cell Type:</label><select id="ctype_select" style="font-size:14px;">{ctype_opts}</select>
    </div>
</div>
"""
js_script = """<script>
    function updateDotplot() {
        var dot_div = document.getElementById("custom_dotplot_div");
        var target_name = "K=" + document.getElementById("k_select").value + "_" + document.getElementById("ctype_select").value;
        var update = {visible: [], 'marker.showscale': []};
        for(var i=0; i<dot_div.data.length; i++) {
            var match = (dot_div.data[i].name === target_name); update.visible.push(match); update['marker.showscale'].push(match);
        } Plotly.restyle(dot_div, update);
    }
    document.getElementById("k_select").addEventListener("change", updateDotplot); document.getElementById("ctype_select").addEventListener("change", updateDotplot);
    setTimeout(updateDotplot, 1000); 
</script>"""

combined_html_body = f"""
<div style="display: flex; flex-direction: column; align-items: center; background-color: white; padding: 20px;">
    {fig_sankey.to_html(include_plotlyjs='cdn', full_html=False)}
    <div style="width: 90%; border-top: 2px dashed #bbb; margin: 40px 0;"></div>
    {fig_spatial.to_html(include_plotlyjs=False, full_html=False)}
    <div style="width: 90%; border-top: 2px dashed #bbb; margin: 40px 0;"></div>
    {html_dot_controls}{fig_dotplot.to_html(include_plotlyjs=False, full_html=False, div_id="custom_dotplot_div")}{js_script}
</div>
"""
save_path = os.path.join(FIG_DIR, "MV1_Step05_Interactive_DualControl_Map_UniqueColor.html")
# Retain the code for HTML outputs excluded from the Final Figure list, without saving/displaying them.
# with open(save_path, "w", encoding="utf-8") as f: f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>CN Evolution & Neighborhood Composition Spot Analysis</title><script src='https://cdn.plot.ly/plotly-latest.min.js'></script></head><body style='margin:0; padding:0; background-color:#f8f9fa;'>{combined_html_body}</body></html>")
# print(f"✅ Color overlap resolved! HTML save path: [ {save_path} ]")
# IPython.display.display(IPython.display.HTML(combined_html_body))


# %%
# ==============================================================================
# [Step 05-7] Differential Dotplot with automatic differential-marker selection
# ▷ Automatically select markers with large between-CN differences using Wilcoxon rank-sum one-vs-rest + FDR correction
# ▷ Generate traces for two modes: all (all markers) + diff (differential markers)
# ▷ Output interactive HTML + JS controls
# ==============================================================================

from scipy import stats
import warnings

print("🔹 [Step 05-7] Generating a Dotplot with automatic Wilcoxon-based differential-marker selection...")

N_TOP_MARKERS = 12  # 🔧 Number of top markers to select (adjustable)

# =====================================================================
# 📌 1. Automatically select markers with large between-CN differences (Wilcoxon one-vs-rest + FDR)
# =====================================================================
def adjust_step05_pvalues(p_values):
    """Adjust Step05 differential-marker screening p-values for multiple testing."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values
    try:
        from statsmodels.stats.multitest import multipletests
        return multipletests(
            p_values,
            alpha=STEP05_DIFF_MARKER_PADJ_ALPHA,
            method=STEP05_DIFF_MARKER_PADJ_METHOD
        )[1]
    except Exception:
        n_tests = len(p_values)
        order = np.argsort(p_values)
        ranked = p_values[order] * n_tests / np.arange(1, n_tests + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted = np.empty_like(ranked)
        adjusted[order] = np.clip(ranked, 0, 1)
        return adjusted


def get_diff_markers(c_expr, cn_label_series, marker_list, n_top=12):
    """Select markers with the largest differences between CN groups by Wilcoxon one-vs-rest effect size."""
    unique_cns = cn_label_series.unique()
    test_records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in marker_list:
            for cn in unique_cns:
                in_mask = (cn_label_series == cn)
                out_mask = ~in_mask
                if in_mask.sum() <= 1 or out_mask.sum() <= 1:
                    continue
                x_in = c_expr.loc[in_mask, m].values
                x_out = c_expr.loc[out_mask, m].values
                try:
                    stat, p_val = stats.ranksums(x_in, x_out)
                    effect = float(np.nanmean(x_in) - np.nanmean(x_out))
                    if np.isfinite(p_val) and np.isfinite(effect):
                        test_records.append((m, cn, effect, p_val))
                except Exception:
                    continue

    marker_scores = {m: 0.0 for m in marker_list}
    marker_best_cn = {m: '' for m in marker_list}
    if test_records:
        adjusted_pvals = adjust_step05_pvalues([r[3] for r in test_records])
        for (m, cn, effect, _), padj in zip(test_records, adjusted_pvals):
            if padj < STEP05_DIFF_MARKER_PADJ_ALPHA:
                score = abs(effect)
                if score > marker_scores[m]:
                    marker_scores[m] = score
                    marker_best_cn[m] = str(cn)

    sorted_markers = sorted(marker_scores, key=lambda x: marker_scores[x], reverse=True)
    significant_markers = [m for m in sorted_markers if marker_scores[m] > 0]
    if not significant_markers:
        significant_markers = sorted_markers
    return significant_markers[:n_top], marker_scores, marker_best_cn


# =====================================================================
# 📌 2. Generate paired traces for all-marker (all) and differential-marker (diff) modes
# =====================================================================
diff_dot_traces = []

for k in K_SELECTED:
    col_k_smooth = f'CN_K{k}_smooth'
    cns = sorted(dlbcl_obs[col_k_smooth].unique(), key=lambda x: int(x.split('_')[1][2:]))

    for c_type in unique_labels:
        type_mask = (dlbcl_obs['Niche_Label'] == c_type)
        c_expr_sub = expr_df[type_mask]
        cn_labels_sub = dlbcl_obs.loc[type_mask, col_k_smooth]

        # Calculate differential markers (Wilcoxon one-vs-rest, ranked by effect size after FDR correction)
        diff_markers, diff_scores, marker_best_cn = get_diff_markers(
            c_expr_sub, cn_labels_sub, markers, n_top=N_TOP_MARKERS
        )

        for mode in ['all', 'diff']:
            mlist = markers if mode == 'all' else diff_markers
            is_visible = bool(
                (k == K_SELECTED[0]) and (c_type == unique_labels[0]) and (mode == 'diff')
            )
            is_first = bool((k == K_SELECTED[0]) and (c_type == unique_labels[0]))

            x_vals, y_vals, colors, sizes, hover_list = [], [], [], [], []
            for cn in cns[::-1]:
                cn_mask = dlbcl_obs.loc[type_mask, col_k_smooth] == cn
                if cn_mask.sum() > 0:
                    cn_expr_s = c_expr_sub[cn_mask]
                    mean_v = cn_expr_s.mean(axis=0)
                    frac_v = (cn_expr_s > 0.0).mean(axis=0)
                else:
                    mean_v = pd.Series(0, index=markers)
                    frac_v = pd.Series(0, index=markers)

                for m in mlist:
                    # diff mode: show ranks (#N) on the X-axis
                    rank_note = f' #{diff_markers.index(m)+1}' if mode == 'diff' else ''
                    x_vals.append(m + rank_note)
                    y_vals.append(cn.split('_')[1])
                    colors.append(float(mean_v[m]))
                    sizes.append(float(frac_v[m]) * 120)
                    hover_txt = (
                        f"<b>{c_type}</b><br>CN: {cn}<br>Marker: {m}"
                        + (f"<br>⚡ Wilcoxon |Δmean| score: {diff_scores.get(m, 0):.2f}  (Rank #{diff_markers.index(m)+1})"
                           f"<br>Best one-vs-rest CN: {marker_best_cn.get(m, '')}"
                           if mode == 'diff' else '')
                        + f"<br>Mean Expr: {mean_v[m]:.3f}<br>Fraction: {frac_v[m]*100:.1f}%"
                    )
                    hover_list.append(hover_txt)

            diff_dot_traces.append(go.Scatter(
                x=x_vals, y=y_vals, mode='markers',
                marker=dict(
                    size=sizes, sizemode='area',
                    color=colors, colorscale='Reds',
                    showscale=(is_first and mode == 'all'),
                    colorbar=dict(title='Mean Expr', x=1.05)
                    if (is_first and mode == 'all') else None,
                    line=dict(color='gray', width=0.5)
                ),
                name=f'K={k}_{c_type}_{mode}',
                visible=is_visible,
                text=hover_list, hoverinfo='text'
            ))

print(f"  ▶ Total traces: {len(diff_dot_traces)} (generated separately for all and diff modes)")

# =====================================================================
# 📌 3. Figure layout
# =====================================================================
fig_diff = go.Figure(data=diff_dot_traces)
fig_diff.update_layout(
    xaxis=dict(
        tickangle=45, showgrid=True, gridcolor='#f0f0f0',
        showline=True, linecolor='black', linewidth=2, mirror=True,
        title='Marker'
    ),
    yaxis=dict(
        showgrid=True, gridcolor='#f0f0f0', automargin=True,
        showline=True, linecolor='black', linewidth=2, mirror=True,
        title='CN Cluster'
    ),
    plot_bgcolor='white',
    width=1500,
    height=420 + (max(K_SELECTED) * 20),
    margin=dict(t=30, b=100, l=80, r=60)
)

# =====================================================================
# 📌 4. Interactive HTML + JS controls
# =====================================================================
k_opts_d     = "".join([f'<option value="{k}">Resolution K = {k}</option>' for k in K_SELECTED])
ctype_opts_d = "".join([f'<option value="{ct}">{ct}</option>' for ct in unique_labels])

diff_html_controls = f"""
<div style="font-family:Arial; width:1500px; text-align:left; margin-bottom:-10px; margin-top:20px;">
  <h3 style="display:inline-block; margin-right:18px;">
    [Step 05-7] Differential Marker Dotplot
    <span style="font-size:12px; color:#888; font-weight:normal;">
      &nbsp;⚡ Automatic selection using Wilcoxon one-vs-rest + FDR (Top {N_TOP_MARKERS})
    </span>
  </h3>
  <div style="display:inline-block; background:#f0f0f0; padding:10px 16px;
              border-radius:6px; border:1px solid #ccc;">
    <label style="font-weight:bold;">📍 K:</label>
    <select id="dK" style="font-size:14px; margin-right:14px;">{k_opts_d}</select>

    <label style="font-weight:bold;">🦠 Cell Type:</label>
    <select id="dCT" style="font-size:14px; margin-right:14px;">{ctype_opts_d}</select>

    <label style="font-weight:bold;">🔬 Displayed markers:</label>
    <select id="dMode" style="font-size:14px; background:#fff8e1;
             border:1px solid #fbc02d; border-radius:4px; padding:2px 6px;">
      <option value="diff" selected>⚡ Differential markers only — Top {N_TOP_MARKERS}</option>
      <option value="all">📋 All markers (All)</option>
    </select>
  </div>
  <div id="dInfo" style="margin-top:6px; font-size:12px; color:#d68910;">
    ⚡ <b>Differential-marker mode:</b> Among markers passing Wilcoxon one-vs-rest, FDR-adjusted p &lt; {STEP05_DIFF_MARKER_PADJ_ALPHA}, show the top {N_TOP_MARKERS} by |Δmean|.
  </div>
</div>
"""

diff_js = """
<script>
  function updateDiff() {
    var el    = document.getElementById('diff_dotplot_div');
    var kv    = document.getElementById('dK').value;
    var ctv   = document.getElementById('dCT').value;
    var modeV = document.getElementById('dMode').value;
    var tgt   = 'K=' + kv + '_' + ctv + '_' + modeV;

    var vis = [], sc = [];
    for (var i = 0; i < el.data.length; i++) {
      var match = (el.data[i].name === tgt);
      vis.push(match); sc.push(match);
    }
    Plotly.restyle(el, {visible: vis, 'marker.showscale': sc});

    var info = document.getElementById('dInfo');
    if (modeV === 'diff') {
      info.innerHTML = '⚡ <b>Differential-marker mode:</b> Show markers passing Wilcoxon one-vs-rest + FDR (X-axis = Marker #rank). Hover to view |Δmean| scores.';
      info.style.color = '#d68910';
    } else {
      info.innerHTML = '📋 <b>All-marker mode:</b> Show every marker in the panel.';
      info.style.color = '#555';
    }
  }
  ['dK','dCT','dMode'].forEach(function(id){
    document.getElementById(id).addEventListener('change', updateDiff);
  });
  setTimeout(updateDiff, 900);
</script>
"""

diff_html_body = f"""
<div style="display:flex; flex-direction:column; align-items:center;
            background:white; padding:20px; font-family:Arial;">
  <h2 style="color:#2c3e50; margin-bottom:4px;">
    📊 Step 05-7 &nbsp;|&nbsp; Differential Marker Dotplot
  </h2>
  <p style="color:#666; font-size:13px; margin-top:0;">
    ⚡ <b>Differential</b> mode: Automatically select markers with large between-CN differences using Wilcoxon one-vs-rest + FDR (Top {N_TOP_MARKERS})<br>
    Dot <b>size</b> = fraction of expressing cells (Fraction),&nbsp; dot <b>color</b> = mean expression (Mean Expr)
  </p>
  {diff_html_controls}
  {fig_diff.to_html(include_plotlyjs=False, full_html=False, div_id='diff_dotplot_div')}
  {diff_js}
</div>
"""

# Retain the HTML-saving code, but exclude it from the Final Figure list.
save_path_diff = os.path.join(FIG_DIR, "MV1_Step05_7_Differential_Dotplot.html")
# with open(save_path_diff, "w", encoding="utf-8") as f:
#     f.write(f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
# <title>Step05-7 Differential Dotplot</title>
# <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
# </head><body style='margin:0;padding:0;background:#f8f9fa;'>{diff_html_body}</body></html>""")
# print(f"✅ [Step 05-7] Complete! HTML saved: {save_path_diff}")
# IPython.display.display(IPython.display.HTML(diff_html_body))

# %%
# ==============================================================================
# [Step 05-8] NMF Neighborhood Analysis
# ▷ Alongside CN analysis, use NMF (Non-negative Matrix Factorization) to
#    extract cellular neighborhood composition topics and assign each cell to its dominant topic.
# ▷ Caution: NMF topics are surrounding cellular composition patterns rather than subtypes.
# ▷ Show spatial mappings of cells (dots) themselves rather than a mosaic.
# ==============================================================================

from sklearn.decomposition import NMF
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os
import math
import IPython.display as display
from matplotlib.lines import Line2D

if True:  # [Change] Always run without skipping to update images and visualizations
    print("\n🔹 [Step 05-8] Running NMF cellular-neighborhood topic analysis across multiple K values...")
    
    K_NMF_LIST = STEP05_NMF_K_LIST
    
    # 🌟 List for storing statistics
    nmf_stats_list = []
    
    for K_NMF in K_NMF_LIST:
        print(f"\n=======================================================")
        print(f" 🚀 Starting NMF analysis (K={K_NMF})")
        print(f"=======================================================")
        
        col_name = f'NMF_K{K_NMF}'
        dlbcl_obs_tmp = adata.obs.loc[cell_ids].copy()

        if STEP05_FIGURE_ONLY:
            if col_name not in adata.obs.columns:
                print(f"  ⚠️ Figure-only mode: column {col_name} is missing; skip the K={K_NMF} NMF figure.")
                continue
            print(f"  ▶ Figure-only mode: regenerate the NMF topic figure using saved {col_name} labels.")
            dlbcl_obs_tmp[col_name] = dlbcl_obs_tmp[col_name].astype(str)
            cn_col_for_h = f'CN_K{K_NMF}' if f'CN_K{K_NMF}' in dlbcl_obs_tmp.columns else None
            if cn_col_for_h is not None:
                h_df = pd.crosstab(
                    dlbcl_obs_tmp[col_name],
                    dlbcl_obs_tmp[cn_col_for_h].astype(str),
                    normalize='index'
                ).fillna(0)
            else:
                h_df = pd.DataFrame(index=sorted(dlbcl_obs_tmp[col_name].dropna().unique()))
            topic_order = [f"NMF_{i+1}" for i in range(K_NMF)]
            h_df = h_df.reindex(index=topic_order, fill_value=0)
            H = h_df.to_numpy(dtype=float)
            heatmap_xticklabels = h_df.columns.tolist()
            recon_err = np.nan
            heatmap_title = f"Saved NMF Topics vs CN Composition [K={K_NMF}]"
        else:
            # 1. Fit the NMF model
            nmf_model = NMF(n_components=K_NMF, init='nndsvd', random_state=42, max_iter=500)
            
            print(f"  ▶ NMF fitting on X_comp ({X_comp.shape[0]} cells, {X_comp.shape[1]} neighborhood labels)...")
            W = nmf_model.fit_transform(X_comp)  # (N_cells, K_NMF)
            H = nmf_model.components_            # (K_NMF, N_neighborhood_labels)
            
            # 2. Calculate statistics (reconstruction error, etc.)
            recon_err = nmf_model.reconstruction_err_
            
            # Assign each cell to the NMF composition topic with the highest weight
            nmf_labels = [f"NMF_{i+1}" for i in np.argmax(W, axis=1)]
            adata.obs.loc[cell_ids, col_name] = nmf_labels
            dlbcl_obs_tmp[col_name] = nmf_labels
            heatmap_xticklabels = unique_labels
            heatmap_title = f"NMF Topics of Cellular Neighborhood Composition [K={K_NMF}]"
        
        # Assigned-cell count statistics
        cell_counts = dlbcl_obs_tmp[col_name].value_counts().to_dict()
        topic_legend_labels = {}
        
        for i in range(K_NMF):
            c_name = f"NMF_{i+1}"
            cell_count = int(cell_counts.get(c_name, 0))
            cell_ratio = cell_count / len(cell_ids) * 100
            if H.shape[1] > 0:
                dominant_neighborhood = str(heatmap_xticklabels[int(np.argmax(H[i]))])
                max_h_weight = float(np.max(H[i]))
            else:
                dominant_neighborhood = 'not_available'
                max_h_weight = np.nan
            topic_legend_labels[c_name] = (
                f"{c_name}: {dominant_neighborhood}-dominant "
                f"({cell_ratio:.1f}%, n={cell_count:,})"
            )
            nmf_stats_list.append({
                'K': K_NMF,
                'NMF_Topic': c_name,
                'Cell_Count': cell_count,
                'Cell_Ratio(%)': np.round(cell_ratio, 2),
                'Dominant_Neighborhood_Label': dominant_neighborhood,
                'Max_H_Weight': np.round(max_h_weight, 3) if np.isfinite(max_h_weight) else np.nan,
                'Recon_Error_Total': np.round(recon_err, 4),
                'Interpretation_Note': STEP05_NMF_TOPIC_NOTE
            })
            
        if np.isfinite(recon_err):
            print(f"  ▶ K={K_NMF} complete (Reconstruction Error: {recon_err:.4f})")
        else:
            print(f"  ▶ K={K_NMF} complete (figure-only regeneration from saved labels)")
        
        # 3. Visualize the NMF H-Matrix (show exact values with annot=True)
        plt.figure(figsize=(10, max(5, K_NMF*0.5)), facecolor='white')
        sns.heatmap(H, xticklabels=heatmap_xticklabels, yticklabels=[f"Topic {i+1} ({cell_counts.get(f'NMF_{i+1}', 0):,} cells)" for i in range(K_NMF)], 
                    cmap='YlOrRd', linewidths=0.5, annot=True, fmt=".2f", annot_kws={"size": 8})
        plt.title(heatmap_title, fontsize=16, fontweight='bold')
        plt.xlabel("Neighborhood composition label used in X_comp", fontsize=12, fontweight='bold')
        plt.ylabel("NMF composition topic (not subtype)", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.tight_layout()
        # plt.savefig(os.path.join(FIG_DIR, f'mv1_Step05-8_NMF_H_Matrix_K{K_NMF}.png'), dpi=300, bbox_inches='tight')
        # Excluded from the Final Figure list; disable only display.
        # plt.show()

        # 4. Spatial mapping of individual cell dots
        print(f"  ▶ Generating the NMF topic spatial map (K={K_NMF})...")
        nmf_palette = sns.color_palette("husl", K_NMF)
        nmf_color_map = {f"NMF_{i+1}": mcolors.to_hex(nmf_palette[i]) for i in range(K_NMF)}
        
        sample_col = 'Run_Tissue_name'
        x_col = next((col for col in ['X_centroid','CenterX','X','x','x_centroid'] if col in adata.obs.columns), None)
        y_col = next((col for col in ['Y_centroid','CenterY','Y','y','y_centroid'] if col in adata.obs.columns), None)
        
        dlbcl_obs = adata.obs.loc[cell_ids].copy()
        unique_samples = dlbcl_obs[sample_col].unique()
        num_samples = len(unique_samples)
        cols = 5 
        rows = math.ceil(num_samples / cols)
        
        # Reserve room on the right for the color-index description.
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3 + 3.4, rows * 3), dpi=300)
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for i, seq_sample in enumerate(unique_samples):
            ax = axes[i]
            sample_mask = (dlbcl_obs[sample_col] == seq_sample)
            sample_obs = dlbcl_obs[sample_mask]
            
            for nmf_lbl in sorted(sample_obs[col_name].unique()):
                cn_mask = (sample_obs[col_name] == nmf_lbl)
                node_color = nmf_color_map.get(nmf_lbl, 'gray')
                
                ax.scatter(sample_obs.loc[cn_mask, x_col], 
                           sample_obs.loc[cn_mask, y_col], 
                           c=node_color, s=2.0, alpha=0.9, edgecolors='none')
                
            ax.set_title(f"Sample: {seq_sample}", fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
            
        for j in range(num_samples, len(axes)):
            axes[j].axis('off')
            
        legend_topics = [f"NMF_{i+1}" for i in range(K_NMF)]
        legend_handles = [
            Line2D(
                [0], [0],
                marker='o',
                linestyle='',
                markersize=5.5,
                markerfacecolor=nmf_color_map.get(topic, 'gray'),
                markeredgecolor='none',
                alpha=0.9,
                label=topic_legend_labels.get(topic, topic)
            )
            for topic in legend_topics
        ]
        fig.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(0.82, 0.46),
            ncol=1,
            title=STEP05_NMF_SPATIAL_LEGEND_TITLE,
            fontsize=8.5,
            title_fontsize=9.5,
            frameon=True
        )
        
        plt.suptitle(f"NMF Topic Spatial Map (K={K_NMF}) - Dot Based\n{STEP05_NMF_TOPIC_NOTE}", fontsize=18, fontweight='bold', y=1.04)
        plt.tight_layout(rect=[0, 0.03, 0.80, 1])
        # plt.savefig(os.path.join(FIG_DIR, f'mv1_Step05-8_NMF_SpatialMap_K{K_NMF}.png'), dpi=300, bbox_inches='tight')
        # Excluded from the Final Figure list; disable only display.
        # plt.show()

    # 5. Compare CN clustering and NMF topics directly using the same cells
    if STEP05_COMPARE_CN_NMF:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        compare_rows = []
        compare_tables = {}
        compare_k_list = [
            k for k in STEP05_CN_NMF_COMPARE_K_LIST
            if f'CN_K{k}' in adata.obs.columns and f'NMF_K{k}' in adata.obs.columns
        ]

        if compare_k_list:
            fig_cmp, axes_cmp = plt.subplots(
                1, len(compare_k_list),
                figsize=(max(6, 5.5 * len(compare_k_list)), 5.5),
                squeeze=False
            )

            for ax_idx, K_COMPARE in enumerate(compare_k_list):
                cn_col = f'CN_K{K_COMPARE}'
                nmf_col = f'NMF_K{K_COMPARE}'
                compare_obs = adata.obs.loc[cell_ids, [cn_col, nmf_col]].dropna()

                if compare_obs.empty:
                    continue

                cn_labels = compare_obs[cn_col].astype(str)
                nmf_topic_labels = compare_obs[nmf_col].astype(str)
                ari = adjusted_rand_score(cn_labels, nmf_topic_labels)
                nmi = normalized_mutual_info_score(cn_labels, nmf_topic_labels)

                overlap_counts = pd.crosstab(cn_labels, nmf_topic_labels)
                overlap_pct = pd.crosstab(cn_labels, nmf_topic_labels, normalize='index') * 100
                compare_tables[K_COMPARE] = overlap_pct

                compare_rows.append({
                    'K': K_COMPARE,
                    'N_cells': int(len(compare_obs)),
                    'Adjusted_Rand_Index': float(ari),
                    'Normalized_Mutual_Info': float(nmi),
                    'CN_N': int(overlap_counts.shape[0]),
                    'NMF_Topic_N': int(overlap_counts.shape[1]),
                    'Interpretation_Note': 'High overlap means CN clustering and NMF topics capture similar neighborhood structure; disagreement suggests complementary structure.'
                })

                ax = axes_cmp[0, ax_idx]
                sns.heatmap(
                    overlap_pct,
                    cmap='YlGnBu',
                    annot=True,
                    fmt='.1f',
                    linewidths=0.5,
                    linecolor='white',
                    cbar=(ax_idx == len(compare_k_list) - 1),
                    cbar_kws={'label': 'CN row % assigned to NMF topic'} if ax_idx == len(compare_k_list) - 1 else None,
                    ax=ax
                )
                ax.set_title(f'CN vs NMF Topic Overlap (K={K_COMPARE})\nARI={ari:.3f}, NMI={nmi:.3f}', fontsize=12, fontweight='bold')
                ax.set_xlabel('NMF composition topic')
                ax.set_ylabel('CN cluster' if ax_idx == 0 else '')

                overlap_pct.to_csv(os.path.join(FINAL_TABLE_DIR, f'mv1_Step05-8B_CN_vs_NMF_Overlap_K{K_COMPARE}.csv'))

            plt.suptitle('CN Clustering vs NMF Cellular-Neighborhood Topic Comparison', fontsize=16, fontweight='bold', y=1.04)
            plt.tight_layout()
            # plt.savefig(os.path.join(FIG_DIR, 'mv1_Step05-8B_CN_vs_NMF_Comparison.png'), dpi=300, bbox_inches='tight')
            # Excluded from the Final Figure list; disable only display.
            # plt.show()

            df_cn_nmf_compare = pd.DataFrame(compare_rows)
            df_cn_nmf_compare.to_csv(os.path.join(FINAL_TABLE_DIR, 'mv1_Step05-8B_CN_vs_NMF_Comparison_Summary.csv'), index=False)
            print("✅ CN vs NMF comparison figure/CSV saved.")
            display.display(df_cn_nmf_compare)
        else:
            print("⚠️ No common K columns are available for CN vs NMF comparison. Comparison requires both CN_K{k} and NMF_K{k}.")

    # 6. Output a combined DataFrame of basic statistics for each K
    print("\n📊 [NMF Basic Statistics Summary (DataFrame)]")
    df_nmf_stats = pd.DataFrame(nmf_stats_list)
    display.display(df_nmf_stats)
    df_nmf_stats.to_csv(os.path.join(FINAL_TABLE_DIR, 'mv1_Step05-8_NMF_Stats_Summary.csv'), index=False)
    print("✅ NMF Neighborhood multi-K search and statistics saving complete.")

# %%
# ==============================================================================
# [Step 05-9] Save the final Step 05 checkpoint
# ▷ Save once CN labels (CN_K{k}), NMF topics (NMF_K{k}), and Niche_Label are all recorded in adata.obs
# ▷ This file is required for Step 06-1 to skip training when loading
# ==============================================================================
if True:  # [Change] Always run to save updated labels even in skip mode
    if 'CHECKPOINT_STEP05' not in dir():
        CHECKPOINT_STEP05 = os.path.join(CKPT_DIR, 'mv1_step05_cn.h5ad')
    adata.write_h5ad(CHECKPOINT_STEP05)
    print(f"✅ [Step 05-9] Step 05 checkpoint saved: {CHECKPOINT_STEP05}")
    pass
    pass
# else:
# #     print(f"⏩ [Step 05-9] step05_skip=True → skip saving (completed results already exist)")
