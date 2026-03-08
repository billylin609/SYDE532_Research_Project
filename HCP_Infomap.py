"""
Community Detection: Infomap + Louvain on HCP SC Consensus Network (82 regions)
================================================================================
Full 82×82 matrix: cortico-cortical + subcortico-cortical + subcortico-subcortical.
Applies thresholding to reveal community structure, and compares methods.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from infomap import Infomap
from enigmatoolbox.datasets import load_sc

# =============================================================================
# Load and build full 82×82 SC matrix
# =============================================================================
sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()

n_ctx  = sc_ctx.shape[0]   # 68 cortical regions
n_sctx = sc_sctx.shape[0]  # 14 subcortical regions
N      = n_ctx + n_sctx    # 82 total regions

# Full 82×82 matrix
# [ ctx-ctx  (68×68) | ctx-sctx  (68×14) ]
# [ sctx-ctx (14×68) | sctx-sctx (14×14) <- zeros ]
sc_full = np.zeros((N, N))
sc_full[:n_ctx, :n_ctx] = sc_ctx
sc_full[:n_ctx, n_ctx:] = sc_sctx.T
sc_full[n_ctx:, :n_ctx] = sc_sctx

labels = list(sc_ctx_labels) + list(sc_sctx_labels)

# =============================================================================
# Region metadata
# =============================================================================
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

def get_lobe(label, idx):
    if idx < n_ctx:
        return lobe_map.get(label[2:], 'Other')
    return 'Subcortical'

def get_hemi(label):
    return 'L' if label.startswith('L') else 'R'

lobes = [get_lobe(labels[i], i) for i in range(N)]
hemis = [get_hemi(l) for l in labels]

lobe_colors = {
    'Frontal':'#e63946','Parietal':'#457b9d','Temporal':'#2a9d8f',
    'Occipital':'#e9c46a','Cingulate':'#f4a261','Insular':'#8338ec',
    'Subcortical':'#6c757d',
}
lobe_order = ['Frontal','Parietal','Temporal','Occipital','Cingulate','Insular','Subcortical']

# =============================================================================
# Community detection helpers
# =============================================================================
def run_infomap(adj, markov_time=1.0):
    """Run Infomap and return module assignments."""
    im = Infomap(seed=42, two_level=True, silent=True, num_trials=100,
                 markov_time=markov_time)
    for i in range(adj.shape[0]):
        for j in range(i+1, adj.shape[0]):
            if adj[i,j] > 0:
                im.add_link(i, j, adj[i,j])
    im.run()
    mods = [0]*adj.shape[0]
    for node in im.nodes:
        mods[node.node_id] = node.module_id
    return mods, im.num_top_modules, im.codelength

def run_louvain(adj, resolution=1.0):
    """Run Louvain and return module assignments."""
    G = nx.Graph()
    for i in range(adj.shape[0]):
        G.add_node(i)
        for j in range(i+1, adj.shape[0]):
            if adj[i,j] > 0:
                G.add_edge(i, j, weight=adj[i,j])
    communities = nx.community.louvain_communities(G, weight='weight',
                                                     resolution=resolution, seed=42)
    mods = [0]*adj.shape[0]
    for idx, comm in enumerate(communities):
        for node in comm:
            mods[node] = idx
    return mods, len(communities)

# =============================================================================
# Approach 1: Infomap with Markov time parameter (full 82×82)
# =============================================================================
print("="*72)
print("APPROACH 1: Infomap with different Markov time parameters (82×82)")
print("="*72)
print("(Higher Markov time = smaller, more granular communities)\n")

for mt in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    mods, n_mods, cl = run_infomap(sc_full, markov_time=mt)
    print(f"  Markov time = {mt:.1f}:  {n_mods:2d} modules, codelength = {cl:.4f}")

# Choose Markov time = 3.0 for a meaningful partition
mods_infomap, n_mods_im, cl_im = run_infomap(sc_full, markov_time=3.0)
print(f"\n  → Using Markov time = 3.0: {n_mods_im} modules")

# =============================================================================
# Approach 2: Threshold then run Infomap
# =============================================================================
print("\n" + "="*72)
print("APPROACH 2: Threshold network, then standard Infomap (82×82)")
print("="*72)

nonzero = sc_full[sc_full > 0]
for pct in [0, 25, 50, 60, 70, 80]:
    thresh = np.percentile(nonzero, pct)
    sc_thresh = sc_full.copy()
    sc_thresh[sc_thresh < thresh] = 0
    density = np.count_nonzero(np.triu(sc_thresh, k=1)) / (N*(N-1)/2) * 100
    mods, n_mods, cl = run_infomap(sc_thresh)
    print(f"  Threshold at {pct}th pct (weight>{thresh:.2f}, density={density:.1f}%): "
          f"{n_mods} modules, CL={cl:.4f}")

# Use 50th percentile threshold
thresh_val = np.percentile(nonzero, 50)
sc_t50 = sc_full.copy()
sc_t50[sc_t50 < thresh_val] = 0
mods_thresh, n_mods_t, cl_t = run_infomap(sc_t50)
print(f"\n  → Using 50th percentile threshold: {n_mods_t} modules")

# =============================================================================
# Approach 3: Louvain (for comparison)
# =============================================================================
print("\n" + "="*72)
print("APPROACH 3: Louvain modularity optimization (82×82)")
print("="*72)

G_full = nx.Graph()
for i in range(N):
    for j in range(i+1, N):
        if sc_full[i,j] > 0:
            G_full.add_edge(i, j, weight=sc_full[i,j])

for res in [0.5, 1.0, 1.5, 2.0]:
    mods, n = run_louvain(sc_full, resolution=res)
    Q = nx.community.modularity(
        G_full,
        [{i for i in range(N) if mods[i]==m} for m in set(mods)],
        weight='weight')
    print(f"  Resolution = {res:.1f}:  {n} modules, Q = {Q:.4f}")

mods_louvain, n_louv = run_louvain(sc_full, resolution=1.0)

# =============================================================================
# Pick best results for visualization
# =============================================================================
modules = mods_infomap
unique_mods = sorted(set(modules))
n_mods = len(unique_mods)

print("\n" + "="*72)
print(f"FINAL PARTITION: Infomap (Markov time=3.0) → {n_mods} modules  [82 regions]")
print("="*72)

for mod in unique_mods:
    members = [i for i in range(N) if modules[i] == mod]
    lobe_dist = {}
    hemi_dist = {'L':0, 'R':0}
    ctx_count  = sum(1 for m in members if m < n_ctx)
    sctx_count = sum(1 for m in members if m >= n_ctx)
    for m in members:
        lobe_dist[lobes[m]] = lobe_dist.get(lobes[m], 0) + 1
        hemi_dist[hemis[m]] += 1
    lobe_str = ", ".join(f"{k}:{v}" for k,v in sorted(lobe_dist.items(), key=lambda x:-x[1]))
    hemi_str = f"L:{hemi_dist['L']}, R:{hemi_dist['R']}"
    print(f"\n  Module {mod} ({len(members)} regions: {ctx_count} ctx, {sctx_count} sctx)  [{lobe_str}] [{hemi_str}]")
    for m in members:
        rtype = 'ctx ' if m < n_ctx else 'sctx'
        print(f"    [{rtype}] {labels[m]:40s} {hemis[m]}  {lobes[m]}")

# =============================================================================
# Export for web tool (full 82×82)
# =============================================================================
pajek_path = './data/sc_network_for_infomap.net'
with open(pajek_path, 'w') as f:
    f.write(f"*Vertices {N}\n")
    for i in range(N):
        f.write(f'{i+1} "{labels[i]}"\n')
    f.write("*Edges\n")
    for i in range(N):
        for j in range(i+1, N):
            if sc_full[i,j] > 0:
                f.write(f"{i+1} {j+1} {sc_full[i,j]:.4f}\n")
print(f"\n✓ Pajek file exported: {pajek_path}")

# =============================================================================
# VISUALIZATION
# =============================================================================
mod_cmap   = plt.cm.Set2 if n_mods <= 8 else plt.cm.tab20
mod_colors = {mod: mod_cmap(i/max(n_mods-1,1)) for i,mod in enumerate(unique_mods)}

strength = sc_full.sum(axis=0)

fig = plt.figure(figsize=(22, 14))
fig.suptitle('Community Detection — HCP SC Consensus Network (82 regions)\n'
             f'Infomap (Markov time=3.0): {n_mods} communities found',
             fontsize=14, fontweight='bold')

gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

# (a) Reordered adjacency matrix
ax = fig.add_subplot(gs[0, 0])
sorted_idx = sorted(range(N), key=lambda x: (modules[x], x >= n_ctx, lobes[x]))
sc_sorted = sc_full[np.ix_(sorted_idx, sorted_idx)]
sc_disp = sc_sorted.copy()
sc_disp[sc_disp == 0] = np.nan
ax.imshow(sc_disp, cmap='YlGnBu', aspect='equal', interpolation='none')
cursor = 0
for mod in unique_mods:
    count = sum(1 for m in modules if m == mod)
    ax.axhline(y=cursor-0.5, color='red', linewidth=1)
    ax.axvline(x=cursor-0.5, color='red', linewidth=1)
    ax.text(-2, cursor+count/2, f'M{mod}', fontsize=7, va='center', ha='right',
            color=mod_colors[mod], fontweight='bold', clip_on=False)
    cursor += count
ax.set_title('(a) SC Matrix by Infomap Module\n(82×82, sorted by module)', fontsize=10, fontweight='bold')

# (b) Force-directed colored by module
ax = fig.add_subplot(gs[0, 1])
G_vis = nx.Graph()
for i in range(N):
    G_vis.add_node(i)
    for j in range(i+1, N):
        if sc_full[i,j] > 0:
            G_vis.add_edge(i, j, weight=sc_full[i,j])
pos = nx.spring_layout(G_vis, weight='weight', k=0.6, iterations=80, seed=42)

for u,v,d in G_vis.edges(data=True):
    same  = modules[u] == modules[v]
    alpha = 0.2 if same else 0.02
    color = mod_colors[modules[u]] if same else '#ccc'
    ax.plot([pos[u][0],pos[v][0]], [pos[u][1],pos[v][1]],
            color=color, alpha=alpha, linewidth=0.4, zorder=0)
sizes = 20 + 150*(strength/strength.max())
for i in range(N):
    marker = 'o' if i < n_ctx else 'D'
    ax.scatter(pos[i][0], pos[i][1], c=[mod_colors[modules[i]]], s=sizes[i],
              marker=marker, edgecolors='white', linewidths=0.3, zorder=3)
ax.set_title('(b) Network by Infomap Module\n(circles=ctx, diamonds=sctx)', fontsize=10, fontweight='bold')
ax.axis('off')

# (c) Circular connectogram by module
ax = fig.add_subplot(gs[0, 2])
sorted_circ = sorted(range(N), key=lambda x: (modules[x], hemis[x], lobes[x]))
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
R = 1.0
xs = R * np.cos(angles)
ys = R * np.sin(angles)
pos_circ = {sorted_circ[k]: (xs[k], ys[k]) for k in range(N)}

upper_weights = [sc_full[i,j] for i in range(N) for j in range(i+1,N) if sc_full[i,j] > 0]
edge_thresh = np.percentile(upper_weights, 85)

for i in range(N):
    for j in range(i+1,N):
        if sc_full[i,j] >= edge_thresh:
            same  = modules[i] == modules[j]
            color = mod_colors[modules[i]] if same else '#aaa'
            alpha = 0.4 if same else 0.08
            ax.plot([pos_circ[i][0], pos_circ[j][0]], [pos_circ[i][1], pos_circ[j][1]],
                    color=color, alpha=alpha, linewidth=0.5, zorder=0)
for k in range(N):
    i = sorted_circ[k]
    marker = 'o' if i < n_ctx else 'D'
    ax.scatter(xs[k], ys[k], c=[mod_colors[modules[i]]],
               s=25+80*(strength[i]/strength.max()),
               marker=marker, edgecolors='white', linewidths=0.3, zorder=3)

cursor = 0
for mod in unique_mods:
    count = sum(1 for m in modules if m == mod)
    mid_angle = angles[cursor + count//2]
    ax.text(1.2*np.cos(mid_angle), 1.2*np.sin(mid_angle), f'M{mod}',
            fontsize=7, ha='center', va='center', fontweight='bold',
            color=mod_colors[mod])
    cursor += count

ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('(c) Connectogram by Module\n(circles=ctx, diamonds=sctx)', fontsize=10, fontweight='bold')

# (d) Module composition by lobe + region type
ax = fig.add_subplot(gs[1, 0])
bar_data = {}
for mod in unique_mods:
    counts = {l: 0 for l in lobe_order}
    for m in [i for i in range(N) if modules[i]==mod]:
        counts[lobes[m]] += 1
    bar_data[mod] = counts
x = np.arange(len(unique_mods))
bottom = np.zeros(len(unique_mods))
for lobe in lobe_order:
    vals = [bar_data[mod][lobe] for mod in unique_mods]
    ax.bar(x, vals, 0.65, bottom=bottom, label=lobe,
           color=lobe_colors.get(lobe, '#999'),
           edgecolor='white', linewidth=0.3)
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels([f'M{m}\n(n={sum(1 for mi in modules if mi==m)})' for m in unique_mods], fontsize=8)
ax.set_ylabel('# Regions')
ax.set_title('(d) Lobe Composition per Module\n(incl. subcortical)', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, ncol=2)

# (e) Infomap vs Louvain comparison
ax = fig.add_subplot(gs[1, 1])
confusion = np.zeros((len(unique_mods), len(set(mods_louvain))))
louv_mods = sorted(set(mods_louvain))
for i in range(N):
    im_idx = unique_mods.index(modules[i])
    lv_idx = louv_mods.index(mods_louvain[i])
    confusion[im_idx, lv_idx] += 1
ax.imshow(confusion, cmap='Blues', aspect='auto')
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        if confusion[i,j] > 0:
            ax.text(j, i, f'{int(confusion[i,j])}', ha='center', va='center', fontsize=8)
ax.set_xlabel('Louvain Module')
ax.set_ylabel('Infomap Module')
ax.set_xticks(range(len(louv_mods)))
ax.set_xticklabels([f'L{m}' for m in louv_mods], fontsize=8)
ax.set_yticks(range(len(unique_mods)))
ax.set_yticklabels([f'M{m}' for m in unique_mods], fontsize=8)
ax.set_title('(e) Infomap vs Louvain Agreement\n(82 regions)', fontsize=10, fontweight='bold')

# (f) Hemisphere + region-type breakdown per module
ax = fig.add_subplot(gs[1, 2])
bar_starts = np.zeros(len(unique_mods))
bar_labels = [f'M{m}' for m in unique_mods]
y_pos = np.arange(len(unique_mods))
for mod_idx, mod in enumerate(unique_mods):
    members = [i for i in range(N) if modules[i] == mod]
    n_L_ctx  = sum(1 for m in members if m < n_ctx  and hemis[m]=='L')
    n_R_ctx  = sum(1 for m in members if m < n_ctx  and hemis[m]=='R')
    n_L_sctx = sum(1 for m in members if m >= n_ctx and hemis[m]=='L')
    n_R_sctx = sum(1 for m in members if m >= n_ctx and hemis[m]=='R')
    ax.barh(mod_idx, n_L_ctx,  color='#457b9d', alpha=0.9)
    ax.barh(mod_idx, n_R_ctx,  left=n_L_ctx, color='#e63946', alpha=0.9)
    ax.barh(mod_idx, n_L_sctx, left=n_L_ctx+n_R_ctx, color='#457b9d', alpha=0.4, hatch='//')
    ax.barh(mod_idx, n_R_sctx, left=n_L_ctx+n_R_ctx+n_L_sctx, color='#e63946', alpha=0.4, hatch='//')

ax.set_yticks(y_pos)
ax.set_yticklabels(bar_labels, fontsize=9)
ax.set_xlabel('# Regions')
ax.set_title('(f) Hemisphere & Type per Module\n(solid=ctx, hatched=sctx; blue=L, red=R)',
             fontsize=10, fontweight='bold')

plt.savefig('./figs/infomap_communities.png', dpi=180, bbox_inches='tight')
print("✓ Figure saved.")
