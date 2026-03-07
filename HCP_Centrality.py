"""
Hub Identification in the HCP SC Consensus Network
====================================================
Traditional + Complex Contagion centrality measures.
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from enigmatoolbox.datasets import load_sc

sc_ctx, sc_ctx_labels, _, _ = load_sc()
labels = list(sc_ctx_labels)
N = sc_ctx.shape[0]

G = nx.Graph()
for i in range(N):
    G.add_node(i, label=labels[i])
for i in range(N):
    for j in range(i+1, N):
        if sc_ctx[i,j] > 0:
            G.add_edge(i, j, weight=sc_ctx[i,j])

lobe_map = {
    'bankssts':'Temporal','caudalanteriorcingulate':'Cingulate',
    'caudalmiddlefrontal':'Frontal','cuneus':'Occipital',
    'entorhinal':'Temporal','fusiform':'Temporal',
    'inferiorparietal':'Parietal','inferiortemporal':'Temporal',
    'isthmuscingulate':'Cingulate','lateraloccipital':'Occipital',
    'lateralorbitofrontal':'Frontal','lingual':'Occipital',
    'medialorbitofrontal':'Frontal','middletemporal':'Temporal',
    'parahippocampal':'Temporal','paracentral':'Frontal',
    'parsopercularis':'Frontal','parsorbitalis':'Frontal',
    'parstriangularis':'Frontal','pericalcarine':'Occipital',
    'postcentral':'Parietal','posteriorcingulate':'Cingulate',
    'precentral':'Frontal','precuneus':'Parietal',
    'rostralanteriorcingulate':'Cingulate','rostralmiddlefrontal':'Frontal',
    'superiorfrontal':'Frontal','superiorparietal':'Parietal',
    'superiortemporal':'Temporal','supramarginal':'Parietal',
    'frontalpole':'Frontal','temporalpole':'Temporal',
    'transversetemporal':'Temporal','insula':'Insular',
}
lobes = [lobe_map.get(l[2:], 'Other') for l in labels]
lobe_colors = {
    'Frontal':'#e63946','Parietal':'#457b9d','Temporal':'#2a9d8f',
    'Occipital':'#e9c46a','Cingulate':'#f4a261','Insular':'#8338ec',
}

# === TRADITIONAL CENTRALITY ===
inv_weight = {(u,v): 1.0/d['weight'] for u,v,d in G.edges(data=True)}
nx.set_edge_attributes(G, {k: {'inv_weight': v} for k,v in inv_weight.items()})

degree_cent = nx.degree_centrality(G)
node_strength = dict(G.degree(weight='weight'))
betweenness_cent = nx.betweenness_centrality(G, weight='inv_weight', normalized=True)
closeness_cent = nx.closeness_centrality(G, distance='inv_weight')
eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')

# === COMPLEX CONTAGION CENTRALITY (Guilbeault & Centola 2021) ===
def complex_closeness_centrality(adj, threshold=2):
    """
    For complex contagion with threshold T, a node activates only when >= T
    of its neighbors are active. We seed with the focal node + its (T-1)
    strongest neighbors (minimal viable seed cluster), then simulate spreading.
    Complex closeness = (# reachable) / (sum of activation steps to reachable).
    """
    binary = (adj > 0).astype(int)
    np.fill_diagonal(binary, 0)
    n = binary.shape[0]
    cc = np.zeros(n)

    for seed in range(n):
        active = np.zeros(n, dtype=bool)
        active[seed] = True
        # Seed cluster: activate (T-1) strongest neighbors
        neighbors = np.where(adj[seed] > 0)[0]
        if len(neighbors) > 0:
            sorted_n = neighbors[np.argsort(adj[seed, neighbors])[::-1]]
            for k in range(min(threshold-1, len(sorted_n))):
                active[sorted_n[k]] = True

        activation_step = np.full(n, np.inf)
        activation_step[active] = 0
        step = 0
        changed = True
        while changed and step < n:
            step += 1
            changed = False
            active_count = binary @ active.astype(int)
            for j in range(n):
                if not active[j] and active_count[j] >= threshold:
                    active[j] = True
                    activation_step[j] = step
                    changed = True

        reachable = (activation_step < np.inf) & (np.arange(n) != seed)
        n_reach = reachable.sum()
        if n_reach > 0:
            cc[seed] = n_reach / activation_step[reachable].sum()
    return cc

def complex_betweenness_centrality(adj, threshold=2):
    """How often a node is newly activated as an intermediate during spreading."""
    binary = (adj > 0).astype(int)
    np.fill_diagonal(binary, 0)
    n = binary.shape[0]
    cb = np.zeros(n)
    for seed in range(n):
        active = np.zeros(n, dtype=bool)
        active[seed] = True
        neighbors = np.where(adj[seed] > 0)[0]
        if len(neighbors) > 0:
            sorted_n = neighbors[np.argsort(adj[seed, neighbors])[::-1]]
            for k in range(min(threshold-1, len(sorted_n))):
                active[sorted_n[k]] = True
        step = 0
        while step < n:
            step += 1
            ac = binary @ active.astype(int)
            newly = (~active) & (ac >= threshold)
            if not newly.any(): break
            cb[newly] += 1
            active = active | newly
    mx = cb.max()
    if mx > 0: cb = cb / mx
    return cb

print("Computing complex centrality (T=2)...")
cc_T2 = complex_closeness_centrality(sc_ctx, 2)
cb_T2 = complex_betweenness_centrality(sc_ctx, 2)
print("Computing complex centrality (T=3)...")
cc_T3 = complex_closeness_centrality(sc_ctx, 3)

# === COMPILE RESULTS ===
df = pd.DataFrame({
    'Region': labels, 'Lobe': lobes,
    'Hemisphere': ['L' if l.startswith('L_') else 'R' for l in labels],
    'Degree': [G.degree(i) for i in range(N)],
    'Degree_Cent': [degree_cent[i] for i in range(N)],
    'Strength': [node_strength[i] for i in range(N)],
    'Betweenness': [betweenness_cent[i] for i in range(N)],
    'Closeness': [closeness_cent[i] for i in range(N)],
    'Eigenvector': [eigenvector_cent[i] for i in range(N)],
    'Complex_Close_T2': cc_T2, 'Complex_Close_T3': cc_T3,
    'Complex_Btwn_T2': cb_T2,
})

centrality_cols = ['Degree_Cent','Strength','Betweenness','Closeness',
                   'Eigenvector','Complex_Close_T2','Complex_Close_T3']
measure_names = {
    'Degree_Cent':'Degree Centrality','Strength':'Node Strength',
    'Betweenness':'Betweenness Centrality','Closeness':'Closeness Centrality',
    'Eigenvector':'Eigenvector Centrality',
    'Complex_Close_T2':'Complex Closeness (T=2)','Complex_Close_T3':'Complex Closeness (T=3)',
}

print("\n" + "="*72)
print("TOP 5 HUBS BY EACH CENTRALITY MEASURE")
print("="*72)
for col in centrality_cols:
    top5 = df.nlargest(5, col)[['Region','Lobe',col]]
    print(f"\n  {measure_names[col]}:")
    for _, row in top5.iterrows():
        print(f"    {row['Region']:40s} {row['Lobe']:12s} {row[col]:.4f}")

# Composite hub score
for col in centrality_cols:
    mn, mx = df[col].min(), df[col].max()
    df[f'{col}_n'] = (df[col]-mn)/(mx-mn) if mx > mn else 0
df['Composite_Hub'] = df[[f'{c}_n' for c in centrality_cols]].mean(axis=1)

print("\n" + "="*72)
print("TOP 10 COMPOSITE HUBS")
print("="*72)
for _, row in df.nlargest(10, 'Composite_Hub').iterrows():
    print(f"  {row['Region']:40s} {row['Lobe']:12s} composite={row['Composite_Hub']:.3f}")

# === VISUALIZATION ===
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Hub Centrality Measures — HCP SC Consensus Network', fontsize=15, fontweight='bold')

plot_measures = [
    ('Degree_Cent','Degree Centrality','How many direct connections'),
    ('Betweenness','Betweenness Centrality','Bridge / gatekeeper role'),
    ('Closeness','Closeness Centrality','Efficiency of reaching all nodes'),
    ('Eigenvector','Eigenvector Centrality','Connected to other important nodes'),
    ('Complex_Close_T2','Complex Closeness (T=2)','Hub for complex contagion, need 2 neighbors'),
    ('Complex_Close_T3','Complex Closeness (T=3)','Hub for complex contagion, need 3 neighbors'),
]

for ax, (col, title, subtitle) in zip(axes.flat, plot_measures):
    vals = df[col].values
    colors = [lobe_colors[l] for l in lobes]
    vmax = vals.max()
    sizes = 30 + 200*(vals/vmax if vmax > 0 else vals)
    ax.scatter(range(N), vals, c=colors, s=sizes, edgecolors='white', linewidths=0.4, alpha=0.85)
    top3 = df.nlargest(3, col).index
    for idx in top3:
        ax.annotate(labels[idx].replace('L_','L·').replace('R_','R·'),
                   (idx, vals[idx]), fontsize=7, ha='center', va='bottom',
                   xytext=(0,5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, lw=0))
    ax.set_title(f'{title}\n({subtitle})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Region Index'); ax.set_ylabel('Centrality Score')
    ax.grid(True, alpha=0.15)

legend_patches = [mpatches.Patch(color=c, label=l) for l,c in lobe_colors.items()]
fig.legend(handles=legend_patches, loc='lower center', ncol=6, fontsize=9, bbox_to_anchor=(0.5,-0.01))
plt.tight_layout()
plt.savefig('./figs/hub_centrality_analysis.png', dpi=180, bbox_inches='tight')

# Correlation matrix
fig2, ax2 = plt.subplots(figsize=(8, 6.5))
corr = df[centrality_cols].corr()
im = ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
short_names = ['Degree','Strength','Betweenness','Closeness','Eigenvector','Cpx Close\nT=2','Cpx Close\nT=3']
ax2.set_xticks(range(len(centrality_cols))); ax2.set_yticks(range(len(centrality_cols)))
ax2.set_xticklabels(short_names, fontsize=8, rotation=30, ha='right')
ax2.set_yticklabels(short_names, fontsize=8)
for i in range(len(centrality_cols)):
    for j in range(len(centrality_cols)):
        ax2.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(corr.iloc[i,j])>0.6 else 'black')
plt.colorbar(im, ax=ax2, label='Pearson r')
ax2.set_title('Correlation Between Centrality Measures', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('./figs/centrality_correlation.png', dpi=180, bbox_inches='tight')

df.to_csv('./data/hub_centrality_table.csv', index=False)
print("\n✓ All outputs saved.")