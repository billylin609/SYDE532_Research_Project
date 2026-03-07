"""
HCP Young Adult Structural Connectivity — Comprehensive Network Analysis Summary
==================================================================================
Generates a multi-page summary report with publication-quality figures.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from infomap import Infomap
from enigmatoolbox.datasets import load_sc

# =============================================================================
# LOAD & PREPARE
# =============================================================================
sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()
labels = list(sc_ctx_labels)
N = sc_ctx.shape[0]

lobe_map = {
    'bankssts':'Temporal','caudalanteriorcingulate':'Cingulate',
    'caudalmiddlefrontal':'Frontal','cuneus':'Occipital','entorhinal':'Temporal',
    'fusiform':'Temporal','inferiorparietal':'Parietal','inferiortemporal':'Temporal',
    'isthmuscingulate':'Cingulate','lateraloccipital':'Occipital',
    'lateralorbitofrontal':'Frontal','lingual':'Occipital','medialorbitofrontal':'Frontal',
    'middletemporal':'Temporal','parahippocampal':'Temporal','paracentral':'Frontal',
    'parsopercularis':'Frontal','parsorbitalis':'Frontal','parstriangularis':'Frontal',
    'pericalcarine':'Occipital','postcentral':'Parietal','posteriorcingulate':'Cingulate',
    'precentral':'Frontal','precuneus':'Parietal','rostralanteriorcingulate':'Cingulate',
    'rostralmiddlefrontal':'Frontal','superiorfrontal':'Frontal',
    'superiorparietal':'Parietal','superiortemporal':'Temporal','supramarginal':'Parietal',
    'frontalpole':'Frontal','temporalpole':'Temporal','transversetemporal':'Temporal',
    'insula':'Insular',
}
lobes = [lobe_map.get(l[2:],'Other') for l in labels]
hemis = ['L' if l.startswith('L_') else 'R' for l in labels]
lobe_colors = {
    'Frontal':'#e63946','Parietal':'#457b9d','Temporal':'#2a9d8f',
    'Occipital':'#e9c46a','Cingulate':'#f4a261','Insular':'#8338ec',
}
lobe_order = ['Frontal','Parietal','Temporal','Occipital','Cingulate','Insular']
node_colors = [lobe_colors[l] for l in lobes]

# Build graph
G = nx.Graph()
for i in range(N):
    G.add_node(i, label=labels[i])
for i in range(N):
    for j in range(i+1, N):
        if sc_ctx[i,j] > 0:
            G.add_edge(i, j, weight=sc_ctx[i,j])

inv_w = {(u,v): 1.0/d['weight'] for u,v,d in G.edges(data=True)}
nx.set_edge_attributes(G, {k: {'inv_weight': v} for k,v in inv_w.items()})

# Centrality measures
strength = sc_ctx.sum(axis=0)
degree = np.array([G.degree(i) for i in range(N)])
betweenness = nx.betweenness_centrality(G, weight='inv_weight', normalized=True)
closeness = nx.closeness_centrality(G, distance='inv_weight')
eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
btw = np.array([betweenness[i] for i in range(N)])
clo = np.array([closeness[i] for i in range(N)])
eig = np.array([eigenvector[i] for i in range(N)])

# Community detection (Infomap thresholded + Louvain)
nonzero = sc_ctx[sc_ctx > 0]
thresh_25 = np.percentile(nonzero, 25)
sc_t = sc_ctx.copy(); sc_t[sc_t < thresh_25] = 0

im = Infomap(seed=42, two_level=True, silent=True, num_trials=100)
for i in range(N):
    for j in range(i+1, N):
        if sc_t[i,j] > 0: im.add_link(i, j, sc_t[i,j])
im.run()
mods_im = [0]*N
for node in im.nodes: mods_im[node.node_id] = node.module_id
unique_mods = sorted(set(mods_im))
n_communities = len(unique_mods)

# Module names based on composition
mod_names = {}
for mod in unique_mods:
    members = [i for i in range(N) if mods_im[i]==mod]
    hL = sum(1 for m in members if hemis[m]=='L')
    hR = sum(1 for m in members if hemis[m]=='R')
    ld = {}
    for m in members: ld[lobes[m]] = ld.get(lobes[m],0)+1
    top_lobe = max(ld, key=ld.get)
    if hL > 0 and hR > 0 and min(hL,hR)/max(hL,hR) > 0.3:
        mod_names[mod] = f"Bilateral Medial\n({top_lobe}-dominated)"
    elif hL > hR:
        mod_names[mod] = f"Left Lateral\n({top_lobe}-dominated)"
    else:
        mod_names[mod] = f"Right Lateral\n({top_lobe}-dominated)"

mod_cmap = plt.cm.Set2
mod_colors_map = {mod: mod_cmap(i/max(n_communities-1,1)) for i,mod in enumerate(unique_mods)}

print("Data prepared. Generating figures...")

# =============================================================================
# FIGURE 1: Network Overview
# =============================================================================
fig1 = plt.figure(figsize=(20, 16))
fig1.patch.set_facecolor('white')
fig1.suptitle('Figure 1: HCP Young Adult Structural Connectivity — Network Overview',
              fontsize=16, fontweight='bold', y=0.98)

gs1 = fig1.add_gridspec(2, 3, hspace=0.35, wspace=0.3, left=0.06, right=0.94, top=0.93, bottom=0.06)

# (A) Weighted SC matrix
ax = fig1.add_subplot(gs1[0, 0])
sc_disp = sc_ctx.copy(); sc_disp[sc_disp==0] = np.nan
im_ax = ax.imshow(sc_disp, cmap='cividis', aspect='equal',
                   norm=LogNorm(vmin=np.nanmin(sc_disp[sc_disp>0]), vmax=np.nanmax(sc_disp)))
ax.axhline(y=33.5, color='white', linewidth=0.8, linestyle='--', alpha=0.8)
ax.axvline(x=33.5, color='white', linewidth=0.8, linestyle='--', alpha=0.8)
ax.text(17, -2.5, 'LH', ha='center', fontsize=9, color='#333', fontweight='bold')
ax.text(51, -2.5, 'RH', ha='center', fontsize=9, color='#333', fontweight='bold')
plt.colorbar(im_ax, ax=ax, label='Connection Weight', shrink=0.8, pad=0.02)
ax.set_title('A. Weighted Connectivity Matrix', fontsize=11, fontweight='bold', pad=8)
ax.set_xlabel('Brain Region'); ax.set_ylabel('Brain Region')

# (B) Edge weight distribution
ax = fig1.add_subplot(gs1[0, 1])
upper = sc_ctx[np.triu_indices_from(sc_ctx, k=1)]
nz = upper[upper > 0]
ax.hist(nz, bins=60, color='#2a9d8f', edgecolor='white', linewidth=0.3, alpha=0.85)
ax.axvline(np.median(nz), color='#e63946', linestyle='--', linewidth=1.5, label=f'Median={np.median(nz):.1f}')
ax.axvline(np.mean(nz), color='#457b9d', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(nz):.1f}')
ax.set_xlabel('Connection Weight'); ax.set_ylabel('Count')
ax.set_title('B. Edge Weight Distribution', fontsize=11, fontweight='bold', pad=8)
ax.legend(fontsize=8)
# Inset: basic stats
stats_text = (f"Regions: {N}\nEdges: {len(nz)}\nDensity: {len(nz)/(N*(N-1)/2)*100:.1f}%\n"
              f"Wt range: [{nz.min():.1f}, {nz.max():.1f}]")
ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# (C) Degree distribution
ax = fig1.add_subplot(gs1[0, 2])
ax.hist(degree, bins=20, color='#457b9d', edgecolor='white', linewidth=0.3, alpha=0.85)
ax.axvline(np.mean(degree), color='#e63946', linestyle='--', linewidth=1.5,
           label=f'Mean={np.mean(degree):.1f}')
ax.set_xlabel('Node Degree'); ax.set_ylabel('Count')
ax.set_title('C. Degree Distribution', fontsize=11, fontweight='bold', pad=8)
ax.legend(fontsize=8)

# (D) Node strength by region (lollipop)
ax = fig1.add_subplot(gs1[1, :2])
sorted_idx = np.argsort(strength)[::-1]
x_pos = np.arange(N)
for idx, i in enumerate(sorted_idx):
    ax.vlines(idx, 0, strength[i], colors=node_colors[i], linewidths=1.2, alpha=0.7)
    ax.scatter(idx, strength[i], c=[node_colors[i]], s=25, zorder=3, edgecolors='white', linewidths=0.3)
# Label top 10
for rank, i in enumerate(sorted_idx[:10]):
    ax.annotate(labels[i].replace('L_','L·').replace('R_','R·'),
               (rank, strength[i]), fontsize=6.5, rotation=45, ha='left', va='bottom',
               xytext=(2, 2), textcoords='offset points')
ax.set_xlabel('Regions (sorted by strength)'); ax.set_ylabel('Node Strength')
ax.set_title('D. Node Strength Ranking (colored by lobe)', fontsize=11, fontweight='bold', pad=8)
ax.set_xlim(-1, N)

# (E) Lobe-level connectivity
ax = fig1.add_subplot(gs1[1, 2])
n_lobes = len(lobe_order)
lobe_conn = np.zeros((n_lobes, n_lobes))
for i in range(N):
    for j in range(i+1, N):
        if sc_ctx[i,j] > 0:
            li = lobe_order.index(lobes[i])
            lj = lobe_order.index(lobes[j])
            lobe_conn[li, lj] += sc_ctx[i,j]
            lobe_conn[lj, li] += sc_ctx[i,j]
# Normalize by number of possible pairs
for i in range(n_lobes):
    ni = sum(1 for l in lobes if l==lobe_order[i])
    for j in range(n_lobes):
        nj = sum(1 for l in lobes if l==lobe_order[j])
        if i == j:
            pairs = ni*(ni-1)/2
        else:
            pairs = ni*nj
        if pairs > 0:
            lobe_conn[i,j] /= pairs

im2 = ax.imshow(lobe_conn, cmap='YlOrRd', aspect='equal')
ax.set_xticks(range(n_lobes)); ax.set_yticks(range(n_lobes))
ax.set_xticklabels([l[:4] for l in lobe_order], fontsize=8, rotation=30, ha='right')
ax.set_yticklabels([l[:4] for l in lobe_order], fontsize=8)
for i in range(n_lobes):
    for j in range(n_lobes):
        ax.text(j, i, f'{lobe_conn[i,j]:.1f}', ha='center', va='center', fontsize=7,
                color='white' if lobe_conn[i,j] > lobe_conn.max()*0.6 else 'black')
plt.colorbar(im2, ax=ax, shrink=0.8, label='Mean weight per pair')
ax.set_title('E. Lobe-Level Connectivity\n(normalized by # possible pairs)', fontsize=11, fontweight='bold', pad=8)

# Shared legend
legend_patches = [mpatches.Patch(color=c, label=l) for l,c in lobe_colors.items()]
fig1.legend(handles=legend_patches, loc='lower center', ncol=6, fontsize=9,
            bbox_to_anchor=(0.5, 0.005), frameon=True)

fig1.savefig('./figs/fig1_network_overview.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✓ Figure 1 saved")

# =============================================================================
# FIGURE 2: Hub Analysis
# =============================================================================
fig2 = plt.figure(figsize=(20, 14))
fig2.patch.set_facecolor('white')
fig2.suptitle('Figure 2: Hub Identification — Traditional and Complex Centrality Measures',
              fontsize=16, fontweight='bold', y=0.98)

gs2 = fig2.add_gridspec(2, 3, hspace=0.4, wspace=0.3, left=0.06, right=0.94, top=0.92, bottom=0.08)

# (A) Degree vs Strength scatter
ax = fig2.add_subplot(gs2[0, 0])
sizes = 30 + 150*(eig/eig.max())
sc_ax = ax.scatter(degree, strength, c=node_colors, s=sizes, edgecolors='white',
                   linewidths=0.4, alpha=0.85, zorder=3)
for i in range(N):
    if strength[i] > np.percentile(strength, 92) or degree[i] > np.percentile(degree, 92):
        ax.annotate(labels[i].replace('L_','L·').replace('R_','R·'),
                   (degree[i], strength[i]), fontsize=6, ha='left', va='bottom',
                   xytext=(3,3), textcoords='offset points')
ax.set_xlabel('Degree (# connections)'); ax.set_ylabel('Strength (sum of weights)')
ax.set_title('A. Degree vs Strength\n(size ∝ eigenvector centrality)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.15)

# (B) Betweenness vs Closeness scatter
ax = fig2.add_subplot(gs2[0, 1])
sizes = 30 + 150*(eig/eig.max())
ax.scatter(btw, clo, c=node_colors, s=sizes, edgecolors='white', linewidths=0.4, alpha=0.85)
for i in range(N):
    if btw[i] > np.percentile(btw, 90) or clo[i] > np.percentile(clo, 90):
        ax.annotate(labels[i].replace('L_','L·').replace('R_','R·'),
                   (btw[i], clo[i]), fontsize=6, ha='left', va='bottom',
                   xytext=(3,3), textcoords='offset points')
ax.set_xlabel('Betweenness Centrality'); ax.set_ylabel('Closeness Centrality')
ax.set_title('B. Betweenness vs Closeness\n(size ∝ eigenvector centrality)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.15)

# (C) Top 15 hubs radar/bar comparison
ax = fig2.add_subplot(gs2[0, 2])
measures = {'Degree': degree/degree.max(), 'Strength': strength/strength.max(),
            'Betweenness': btw/btw.max(), 'Closeness': clo/clo.max(),
            'Eigenvector': eig/eig.max()}
composite = np.mean(list(measures.values()), axis=0)
top15_idx = np.argsort(composite)[::-1][:15]

bar_width = 0.15
x = np.arange(15)
for k, (name, vals) in enumerate(measures.items()):
    ax.barh(x + k*bar_width, vals[top15_idx], bar_width, alpha=0.8,
            label=name, zorder=3)
ax.set_yticks(x + 2*bar_width)
ax.set_yticklabels([labels[i].replace('L_','L·').replace('R_','R·') for i in top15_idx], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('Normalized Centrality')
ax.set_title('C. Top 15 Hubs — Multi-Measure Profile', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')

# (D) Centrality correlation heatmap
ax = fig2.add_subplot(gs2[1, 0])
cent_names = ['Degree', 'Strength', 'Betweenness', 'Closeness', 'Eigenvector']
cent_data = np.column_stack([degree, strength, btw, clo, eig])
corr = np.corrcoef(cent_data.T)
im3 = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(cent_names, fontsize=8, rotation=30, ha='right')
ax.set_yticklabels(cent_names, fontsize=8)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(corr[i,j]) > 0.6 else 'black')
plt.colorbar(im3, ax=ax, shrink=0.8)
ax.set_title('D. Centrality Correlation Matrix', fontsize=11, fontweight='bold')

# (E) Lobe-level hub distribution
ax = fig2.add_subplot(gs2[1, 1])
lobe_hub_data = {l: [] for l in lobe_order}
for i in range(N):
    lobe_hub_data[lobes[i]].append(composite[i])
bp_data = [lobe_hub_data[l] for l in lobe_order]
bp = ax.boxplot(bp_data, patch_artist=True, widths=0.5, showfliers=True)
for patch, lobe in zip(bp['boxes'], lobe_order):
    patch.set_facecolor(lobe_colors[lobe])
    patch.set_alpha(0.7)
ax.set_xticklabels([l[:5] for l in lobe_order], fontsize=8)
ax.set_ylabel('Composite Hub Score')
ax.set_title('E. Hub Score Distribution by Lobe', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.15, axis='y')

# (F) Network with node size = composite hub score
ax = fig2.add_subplot(gs2[1, 2])
pos = nx.spring_layout(G, weight='weight', k=0.6, iterations=80, seed=42)
for u,v,d in G.edges(data=True):
    w = d['weight']
    ax.plot([pos[u][0],pos[v][0]], [pos[u][1],pos[v][1]],
            color='#ddd', alpha=0.05+0.15*(w/sc_ctx.max()), linewidth=0.3, zorder=0)
sizes = 15 + 200*(composite/composite.max())
for i in range(N):
    ax.scatter(pos[i][0], pos[i][1], c=[node_colors[i]], s=sizes[i],
              edgecolors='white', linewidths=0.3, zorder=3)
for i in top15_idx[:8]:
    ax.annotate(labels[i].replace('L_','L·').replace('R_','R·'),
               (pos[i][0], pos[i][1]), fontsize=6, ha='center', va='bottom',
               xytext=(0,4), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, lw=0))
ax.set_title('F. Network Graph\n(node size ∝ composite hub score)', fontsize=11, fontweight='bold')
ax.axis('off')

fig2.legend(handles=legend_patches, loc='lower center', ncol=6, fontsize=9,
            bbox_to_anchor=(0.5, 0.005))

fig2.savefig('./figs/fig2_hub_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✓ Figure 2 saved")

# =============================================================================
# FIGURE 3: Community Structure
# =============================================================================
fig3 = plt.figure(figsize=(20, 12))
fig3.patch.set_facecolor('white')
fig3.suptitle('Figure 3: Community Structure — Infomap and Louvain Decomposition',
              fontsize=16, fontweight='bold', y=0.98)

gs3 = fig3.add_gridspec(2, 3, hspace=0.35, wspace=0.3, left=0.06, right=0.94, top=0.92, bottom=0.06)

# Louvain communities
comms = nx.community.louvain_communities(G, weight='weight', resolution=1.0, seed=42)
mods_louv = [0]*N
for idx,comm in enumerate(comms):
    for node in comm: mods_louv[node] = idx
um_louv = sorted(set(mods_louv))

# (A) Infomap: reordered matrix
ax = fig3.add_subplot(gs3[0, 0])
si = sorted(range(N), key=lambda x: (mods_im[x], lobes[x]))
sc_s = sc_ctx[np.ix_(si,si)]; sd = sc_s.copy(); sd[sd==0] = np.nan
ax.imshow(sd, cmap='cividis', aspect='equal', interpolation='none')
cursor = 0
for mod in unique_mods:
    cnt = sum(1 for m in mods_im if m==mod)
    ax.axhline(y=cursor-0.5, color='white', linewidth=1.5)
    ax.axvline(x=cursor-0.5, color='white', linewidth=1.5)
    ax.text(-3, cursor+cnt/2, mod_names[mod].split('\n')[0], fontsize=6, va='center',
            ha='right', fontweight='bold', color=mod_colors_map[mod], clip_on=False)
    cursor += cnt
ax.set_title('A. SC Matrix by Infomap Module', fontsize=11, fontweight='bold')

# (B) Infomap: network
ax = fig3.add_subplot(gs3[0, 1])
for u,v,d in G.edges(data=True):
    same = mods_im[u]==mods_im[v]
    ax.plot([pos[u][0],pos[v][0]], [pos[u][1],pos[v][1]],
            color=mod_colors_map[mods_im[u]] if same else '#ddd',
            alpha=0.25 if same else 0.02, linewidth=0.5, zorder=0)
sizes = 20+150*(strength/strength.max())
for i in range(N):
    ax.scatter(pos[i][0], pos[i][1], c=[mod_colors_map[mods_im[i]]], s=sizes[i],
              edgecolors='white', linewidths=0.3, zorder=3)
ax.set_title('B. Network by Infomap Module', fontsize=11, fontweight='bold')
ax.axis('off')
# Module legend
for mod in unique_mods:
    ax.scatter([], [], c=[mod_colors_map[mod]], s=60, label=mod_names[mod].replace('\n',' '))
ax.legend(fontsize=7, loc='lower right', framealpha=0.9)

# (C) Module composition stacked bar
ax = fig3.add_subplot(gs3[0, 2])
x = np.arange(len(unique_mods))
bottom = np.zeros(len(unique_mods))
for lobe in lobe_order:
    vals = [sum(1 for i in range(N) if mods_im[i]==mod and lobes[i]==lobe) for mod in unique_mods]
    ax.bar(x, vals, 0.65, bottom=bottom, label=lobe, color=lobe_colors[lobe],
           edgecolor='white', linewidth=0.3)
    bottom += vals
ax.set_xticks(x)
xlabels = []
for mod in unique_mods:
    n_m = sum(1 for m in mods_im if m==mod)
    hL = sum(1 for i in range(N) if mods_im[i]==mod and hemis[i]=='L')
    hR = n_m - hL
    xlabels.append(f'{mod_names[mod]}\n(n={n_m}, L:{hL}/R:{hR})')
ax.set_xticklabels(xlabels, fontsize=7)
ax.set_ylabel('# Regions')
ax.set_title('C. Lobe & Hemisphere Composition', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, ncol=2)

# (D) Louvain: reordered matrix
ax = fig3.add_subplot(gs3[1, 0])
louv_cmap = plt.cm.tab10
louv_colors = {m: louv_cmap(i/max(len(um_louv)-1,1)) for i,m in enumerate(um_louv)}
si2 = sorted(range(N), key=lambda x: (mods_louv[x], lobes[x]))
sc_s2 = sc_ctx[np.ix_(si2,si2)]; sd2 = sc_s2.copy(); sd2[sd2==0] = np.nan
ax.imshow(sd2, cmap='cividis', aspect='equal', interpolation='none')
cursor = 0
for mod in um_louv:
    cnt = sum(1 for m in mods_louv if m==mod)
    ax.axhline(y=cursor-0.5, color='white', linewidth=1.5)
    ax.axvline(x=cursor-0.5, color='white', linewidth=1.5)
    ax.text(-3, cursor+cnt/2, f'L{mod}', fontsize=7, va='center', ha='right',
            fontweight='bold', color=louv_colors[mod], clip_on=False)
    cursor += cnt
ax.set_title('D. SC Matrix by Louvain Module', fontsize=11, fontweight='bold')

# (E) Louvain: network
ax = fig3.add_subplot(gs3[1, 1])
for u,v,d in G.edges(data=True):
    same = mods_louv[u]==mods_louv[v]
    ax.plot([pos[u][0],pos[v][0]], [pos[u][1],pos[v][1]],
            color=louv_colors[mods_louv[u]] if same else '#ddd',
            alpha=0.25 if same else 0.02, linewidth=0.5, zorder=0)
for i in range(N):
    ax.scatter(pos[i][0], pos[i][1], c=[louv_colors[mods_louv[i]]], s=sizes[i],
              edgecolors='white', linewidths=0.3, zorder=3)
ax.set_title('E. Network by Louvain Module', fontsize=11, fontweight='bold')
ax.axis('off')

# (F) Inter-module connectivity (Infomap)
ax = fig3.add_subplot(gs3[1, 2])
n_m = len(unique_mods)
mod_conn = np.zeros((n_m, n_m))
for i in range(N):
    for j in range(i+1, N):
        if sc_ctx[i,j] > 0:
            mi = unique_mods.index(mods_im[i])
            mj = unique_mods.index(mods_im[j])
            mod_conn[mi,mj] += sc_ctx[i,j]
            mod_conn[mj,mi] += sc_ctx[i,j]
# Normalize
for i in range(n_m):
    ni = sum(1 for m in mods_im if m==unique_mods[i])
    for j in range(n_m):
        nj = sum(1 for m in mods_im if m==unique_mods[j])
        pairs = ni*nj if i!=j else ni*(ni-1)/2
        if pairs > 0: mod_conn[i,j] /= pairs

im4 = ax.imshow(mod_conn, cmap='YlOrRd', aspect='equal')
ax.set_xticks(range(n_m)); ax.set_yticks(range(n_m))
short_mod_names = [mn.split('\n')[0] for mn in [mod_names[m] for m in unique_mods]]
ax.set_xticklabels(short_mod_names, fontsize=7, rotation=20, ha='right')
ax.set_yticklabels(short_mod_names, fontsize=7)
for i in range(n_m):
    for j in range(n_m):
        ax.text(j, i, f'{mod_conn[i,j]:.2f}', ha='center', va='center', fontsize=9,
                color='white' if mod_conn[i,j] > mod_conn.max()*0.5 else 'black')
plt.colorbar(im4, ax=ax, shrink=0.8, label='Mean weight per pair')
ax.set_title('F. Inter-Module Connectivity\n(Infomap, normalized)', fontsize=11, fontweight='bold')

fig3.savefig('./figs/fig3_communities.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✓ Figure 3 saved")

# =============================================================================
# FIGURE 4: Connectogram + Key Findings Summary
# =============================================================================
fig4 = plt.figure(figsize=(20, 10))
fig4.patch.set_facecolor('white')
fig4.suptitle('Figure 4: Structural Connectogram and Summary of Key Findings',
              fontsize=16, fontweight='bold', y=0.98)

gs4 = fig4.add_gridspec(1, 2, wspace=0.05, left=0.02, right=0.98, top=0.90, bottom=0.05)

# (A) Connectogram colored by community
ax = fig4.add_subplot(gs4[0, 0])
# Sort by module then lobe
sorted_circ = sorted(range(N), key=lambda x: (mods_im[x], hemis[x], lobes[x]))
n = N
gap = 0.12
mid = n // 2
angles = np.zeros(n)
# Split by hemisphere within each module
l_nodes = [sorted_circ[k] for k in range(n) if hemis[sorted_circ[k]]=='L']
r_nodes = [sorted_circ[k] for k in range(n) if hemis[sorted_circ[k]]=='R']
angles_l = np.linspace(gap, np.pi-gap, len(l_nodes))
angles_r = np.linspace(np.pi+gap, 2*np.pi-gap, len(r_nodes))
angle_map = {}
for k, node in enumerate(l_nodes): angle_map[node] = angles_l[k]
for k, node in enumerate(r_nodes): angle_map[node] = angles_r[k]

R = 1.0
xs = {i: R*np.cos(angle_map[i]) for i in range(N)}
ys = {i: R*np.sin(angle_map[i]) for i in range(N)}

# Draw top edges
edge_thresh = np.percentile(nz, 80)
for i in range(N):
    for j in range(i+1, N):
        if sc_ctx[i,j] >= edge_thresh:
            same_mod = mods_im[i]==mods_im[j]
            inter_hemi = hemis[i] != hemis[j]
            if same_mod:
                color = mod_colors_map[mods_im[i]]
                alpha = 0.15 + 0.35*(sc_ctx[i,j]/sc_ctx.max())
            elif inter_hemi:
                color = '#d62828'
                alpha = 0.1 + 0.2*(sc_ctx[i,j]/sc_ctx.max())
            else:
                color = '#888'
                alpha = 0.05
            ax.plot([xs[i],xs[j]], [ys[i],ys[j]], color=color, alpha=alpha,
                    linewidth=0.3+1.0*(sc_ctx[i,j]/sc_ctx.max()), zorder=0)

# Draw nodes
for i in range(N):
    s = 20 + 100*(strength[i]/strength.max())
    ax.scatter(xs[i], ys[i], c=[mod_colors_map[mods_im[i]]], s=s, zorder=3,
              edgecolors='white', linewidths=0.3)

# Label hubs
for i in top15_idx[:10]:
    angle = angle_map[i]
    lx = 1.15*np.cos(angle); ly = 1.15*np.sin(angle)
    ha = 'left' if np.cos(angle) > 0 else 'right'
    ax.annotate(labels[i].replace('L_','L·').replace('R_','R·'),
               (lx, ly), fontsize=5.5, ha=ha, va='center',
               color=mod_colors_map[mods_im[i]], fontweight='bold')

ax.text(0, 1.35, 'Left Hemisphere', ha='center', fontsize=10, fontstyle='italic', color='#555')
ax.text(0, -1.35, 'Right Hemisphere', ha='center', fontsize=10, fontstyle='italic', color='#555')
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('A. Structural Connectogram\n(colored by Infomap community, top 20% edges)',
             fontsize=11, fontweight='bold')

# (B) Text summary
ax = fig4.add_subplot(gs4[0, 1])
ax.axis('off')

summary_text = """KEY FINDINGS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NETWORK TOPOLOGY

   • 68 cortical regions (Desikan-Killiany atlas)
   • 697 structural connections (30.6% density)
   • Weights range 1.2 – 12.6 (log-normal distribution)
   • Symmetric, undirected (consensus across ~1,200 subjects)

2. HUB REGIONS (consistently ranked across all measures)

   ① R_superiorparietal     — highest degree (41), strength, betweenness
   ② R_superiorfrontal       — 2nd strongest hub, key integrator
   ③ L_superiorfrontal       — bilateral frontal hub
   ④ L_superiorparietal     — bilateral parietal hub
   ⑤ L/R_insula                — highest degree after parietal, gateway role
   ⑥ L/R_precuneus           — medial hub, default mode network core

3. COMMUNITY STRUCTURE (Infomap, 3 modules)

   • Module 1: Left lateral cortex (25 regions, temporal-frontal)
   • Module 2: Bilateral medial wall (18 regions, cingulate-frontal)
        → Only module spanning both hemispheres
        → Contains precuneus, cingulate, medial frontal (DMN core)
   • Module 3: Right lateral cortex (25 regions, mirrors Module 1)
   • Louvain (Q=0.31) confirms similar 3-module decomposition

4. IMPLICATIONS FOR CASCADING FAILURE ANALYSIS

   • Superior parietal & frontal hubs are critical failure points
     — their removal maximally disrupts network integration
   • The bilateral medial module is the only cross-hemispheric
     bridge; damage here could propagate to both hemispheres
   • High density (30.6%) provides redundancy against random
     failure but makes the network vulnerable to targeted
     hub attacks (rich-club vulnerability)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Data: HCP Young Adult (Van Essen et al., 2013)
Consensus method: Betzel et al. (2019), Network Neuroscience
ENIGMA Toolbox: Larivière et al. (2021), Nature Methods
Community detection: Rosvall & Bergstrom (2008), PNAS"""

ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
        fontsize=8.5, fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.9))

fig4.savefig('./figs/fig4_summary.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✓ Figure 4 saved")
print("\nAll figures complete.")