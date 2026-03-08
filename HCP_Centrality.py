"""
Hub Identification in the HCP SC Consensus Network (82 regions)
================================================================
Node-level:   Degree, Betweenness, Closeness, Eigenvector, Eigenvalue (subgraph)
Global-level: Density, Global Efficiency, Avg Clustering, Transitivity,
              Spectral Radius, Algebraic Connectivity, Avg Path Length
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from enigmatoolbox.datasets import load_sc

# =============================================================================
# Load and build full 82×82 SC matrix
# =============================================================================
sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()

n_ctx  = sc_ctx.shape[0]
n_sctx = sc_sctx.shape[0]
N      = n_ctx + n_sctx

sc_full = np.zeros((N, N))
sc_full[:n_ctx, :n_ctx] = sc_ctx
sc_full[:n_ctx, n_ctx:] = sc_sctx.T
sc_full[n_ctx:, :n_ctx] = sc_sctx

labels = list(sc_ctx_labels) + list(sc_sctx_labels)

G = nx.Graph()
for i in range(N):
    G.add_node(i, label=labels[i])
for i in range(N):
    for j in range(i+1, N):
        if sc_full[i, j] > 0:
            G.add_edge(i, j, weight=sc_full[i, j])

# =============================================================================
# Region metadata
# =============================================================================
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

def get_lobe(label, idx):
    return lobe_map.get(label[2:], 'Other') if idx < n_ctx else 'Subcortical'

def get_hemi(label):
    return 'L' if label.startswith('L') else 'R'

lobes = [get_lobe(labels[i], i) for i in range(N)]
hemis = [get_hemi(l) for l in labels]

lobe_colors = {
    'Frontal':'#e63946','Parietal':'#457b9d','Temporal':'#2a9d8f',
    'Occipital':'#e9c46a','Cingulate':'#f4a261','Insular':'#8338ec',
    'Subcortical':'#6c757d',
}

# =============================================================================
# NODE-LEVEL CENTRALITY
# =============================================================================
inv_weight = {(u,v): 1.0/d['weight'] for u,v,d in G.edges(data=True)}
nx.set_edge_attributes(G, {k: {'inv_weight': v} for k,v in inv_weight.items()})

print("Computing node centrality measures...")
degree_cent      = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, weight='inv_weight', normalized=True)
closeness_cent   = nx.closeness_centrality(G, distance='inv_weight')
eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')

# PageRank (Eigenvalue) centrality — equation from image:
#   x_i = α · Σ_j A_ij · (x_j / k_j^out) + β
#   α = damping factor (0.85), β = (1-α)/N  (teleportation)
#   For undirected SC: k_j^out = weighted degree of j
print("Computing PageRank (eigenvalue) centrality  [α=0.85]...")
pagerank_cent = nx.pagerank(G, alpha=0.85, weight='weight')
pagerank_arr  = np.array([pagerank_cent[i] for i in range(N)])

# Adjacency matrix for spectral metrics
A        = nx.to_numpy_array(G, weight='weight')
eigvals_A = np.linalg.eigvalsh(A)

# =============================================================================
# GLOBAL CENTRALITY METRICS
# =============================================================================
print("Computing global centrality & network metrics...")

# ── Global node-centrality aggregates ────────────────────────────────────────
# C_D^{global} = (1/N) Σ_i C_D(i)
global_degree_cent      = np.mean(list(degree_cent.values()))
# C_B^{global} = (1/N) Σ_i C_B(i)
global_betweenness_cent = np.mean(list(betweenness_cent.values()))
# C_C^{global} = (1/N) Σ_i C_C(i)
global_closeness_cent   = np.mean(list(closeness_cent.values()))
# C_EV^{global} = (1/N) Σ_i C_EV(i)
global_eigenvector_cent = np.mean(list(eigenvector_cent.values()))
# C_PR^{global} = (1/N) Σ_i x_i  (= 1/N by construction, shown for completeness)
global_pagerank_cent    = np.mean(pagerank_arr)

# ── Spectral / topological network metrics ────────────────────────────────────
spectral_radius       = float(eigvals_A.max())
spectral_gap          = float(eigvals_A[-1] - eigvals_A[-2])

L_mat  = nx.laplacian_matrix(G).toarray().astype(float)
lap_eigs = np.linalg.eigvalsh(L_mat)
algebraic_connectivity = float(lap_eigs[1])

density           = nx.density(G)
global_efficiency = nx.global_efficiency(G)
avg_clustering    = nx.average_clustering(G, weight='weight')
transitivity      = nx.transitivity(G)
avg_path_length   = nx.average_shortest_path_length(G, weight='inv_weight') \
                    if nx.is_connected(G) else float('nan')

global_metrics = {
    # Node-centrality aggregates
    'Global Degree Centrality':      global_degree_cent,
    'Global Betweenness Centrality': global_betweenness_cent,
    'Global Closeness Centrality':   global_closeness_cent,
    'Global Eigenvector Centrality': global_eigenvector_cent,
    'Global PageRank Centrality':    global_pagerank_cent,
    # Topological
    'Density':                       density,
    'Global Efficiency':             global_efficiency,
    'Avg Clustering Coeff':          avg_clustering,
    'Transitivity':                  transitivity,
    'Avg Path Length (wt)':          avg_path_length,
    # Spectral
    'Spectral Radius (λ_max)':       spectral_radius,
    'Spectral Gap':                  spectral_gap,
    'Algebraic Connectivity':        algebraic_connectivity,
}

print("\n" + "="*60)
print("GLOBAL CENTRALITY & NETWORK METRICS  (82 regions)")
print("="*60)
for k, v in global_metrics.items():
    print(f"  {k:<35s}: {v:.4f}")

# =============================================================================
# COMPILE NODE-LEVEL RESULTS
# =============================================================================
centrality_cols = ['Degree_Cent', 'Betweenness', 'Closeness', 'Eigenvector', 'PageRank']
measure_names   = {
    'Degree_Cent': 'Degree Centrality',
    'Betweenness': 'Betweenness Centrality',
    'Closeness':   'Closeness Centrality',
    'Eigenvector': 'Eigenvector Centrality',
    'PageRank':    'PageRank (Eigenvalue) Centrality',
}

df = pd.DataFrame({
    'Region':      labels,
    'Lobe':        lobes,
    'Hemisphere':  hemis,
    'Region_Type': ['Cortical']*n_ctx + ['Subcortical']*n_sctx,
    'Degree':      [G.degree(i) for i in range(N)],
    'Degree_Cent': [degree_cent[i]      for i in range(N)],
    'Strength':    [dict(G.degree(weight='weight'))[i] for i in range(N)],
    'Betweenness': [betweenness_cent[i] for i in range(N)],
    'Closeness':   [closeness_cent[i]   for i in range(N)],
    'Eigenvector': [eigenvector_cent[i] for i in range(N)],
    'PageRank':    pagerank_arr,
})

print("\n" + "="*72)
print("TOP 5 HUBS BY EACH CENTRALITY MEASURE  (all 82 regions)")
print("="*72)
for col in centrality_cols:
    top5 = df.nlargest(5, col)[['Region', 'Region_Type', 'Lobe', col]]
    print(f"\n  {measure_names[col]}:")
    for _, row in top5.iterrows():
        print(f"    {row['Region']:40s} [{row['Region_Type']:11s}] {row['Lobe']:12s} {row[col]:.4f}")

# Composite hub score
for col in centrality_cols:
    mn, mx = df[col].min(), df[col].max()
    df[f'{col}_n'] = (df[col]-mn)/(mx-mn) if mx > mn else 0
df['Composite_Hub'] = df[[f'{c}_n' for c in centrality_cols]].mean(axis=1)

print("\n" + "="*72)
print("TOP 10 COMPOSITE HUBS")
print("="*72)
for _, row in df.nlargest(10, 'Composite_Hub').iterrows():
    print(f"  {row['Region']:40s} [{row['Region_Type']:11s}] {row['Lobe']:12s} "
          f"composite={row['Composite_Hub']:.3f}")

# =============================================================================
# FIGURE 1: Node-level centrality — single row, 5 panels
# =============================================================================
node_colors = [lobe_colors.get(l, '#999') for l in lobes]
ctx_mask    = np.array([i < n_ctx  for i in range(N)])
sctx_mask   = ~ctx_mask

plot_measures = [
    ('Degree_Cent', 'Degree Centrality',              '# direct connections'),
    ('Betweenness', 'Betweenness Centrality',          'bridge / gatekeeper role'),
    ('Closeness',   'Closeness Centrality',             'efficiency of reaching all nodes'),
    ('Eigenvector', 'Eigenvector Centrality',           'connected to other high-degree nodes'),
    ('PageRank',    'PageRank (Eigenvalue) Centrality','α·Σ A_ij·(x_j/k_j^out) + β'),
]

fig, axes = plt.subplots(1, 5, figsize=(26, 5))
fig.suptitle('Node Centrality Measures — HCP SC Consensus Network (82 regions)\n'
             '○ cortical  ◆ subcortical', fontsize=13, fontweight='bold')

for ax, (col, title, subtitle) in zip(axes, plot_measures):
    vals  = df[col].values
    vmax  = vals.max()
    sizes = 25 + 180*(vals / vmax if vmax > 0 else vals)

    ax.scatter(np.where(ctx_mask)[0],  vals[ctx_mask],
               c=[node_colors[i] for i in np.where(ctx_mask)[0]],
               s=sizes[ctx_mask],  marker='o',
               edgecolors='white', linewidths=0.4, alpha=0.85)
    ax.scatter(np.where(sctx_mask)[0], vals[sctx_mask],
               c=[node_colors[i] for i in np.where(sctx_mask)[0]],
               s=sizes[sctx_mask], marker='D',
               edgecolors='white', linewidths=0.4, alpha=0.85)

    ax.axvline(x=n_ctx - 0.5, color='gray', linewidth=1, linestyle='--', alpha=0.4)
    ax.text(n_ctx/2,        vals.max()*1.02, 'Cortical',    ha='center', fontsize=7, color='gray')
    ax.text(n_ctx+n_sctx/2, vals.max()*1.02, 'Subcortical', ha='center', fontsize=7, color='gray')

    for idx in df.nlargest(3, col).index:
        ax.annotate(labels[idx].replace('L_','L·').replace('R_','R·'),
                    (idx, vals[idx]), fontsize=6.5, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, lw=0))

    ax.set_title(f'{title}\n({subtitle})', fontsize=9, fontweight='bold')
    ax.set_xlabel('Region index', fontsize=8)
    ax.set_ylabel('Score', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.12)

legend_patches = [mpatches.Patch(color=c, label=l) for l, c in lobe_colors.items()]
fig.legend(handles=legend_patches, loc='lower center', ncol=7, fontsize=8,
           bbox_to_anchor=(0.5, -0.06))
plt.tight_layout()
plt.savefig('./figs/hub_centrality_analysis.png', dpi=180, bbox_inches='tight')
print("\n✓ hub_centrality_analysis.png saved")

# =============================================================================
# FIGURE 2: Global metrics bar chart
# =============================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Global Network Metrics — HCP SC Consensus Network (82 regions)',
              fontsize=13, fontweight='bold')

# Left: topological metrics (0–1 range)
topo_keys = ['Density', 'Global Efficiency', 'Avg Clustering Coeff', 'Transitivity']
topo_vals = [global_metrics[k] for k in topo_keys]
bars = axes2[0].barh(topo_keys, topo_vals,
                     color=['#457b9d','#2a9d8f','#e9c46a','#f4a261'],
                     edgecolor='white', alpha=0.88)
for bar, v in zip(bars, topo_vals):
    axes2[0].text(v + 0.005, bar.get_y() + bar.get_height()/2,
                  f'{v:.4f}', va='center', fontsize=9)
axes2[0].set_xlim(0, max(topo_vals)*1.25)
axes2[0].set_title('Topological Metrics', fontsize=11, fontweight='bold')
axes2[0].set_xlabel('Value')
axes2[0].grid(True, alpha=0.15, axis='x')

# Right: spectral + path metrics
spec_keys = ['Spectral Radius (λ_max)', 'Spectral Gap',
             'Algebraic Connectivity', 'Avg Path Length (wt)']
spec_vals = [global_metrics[k] for k in spec_keys]
bars2 = axes2[1].barh(spec_keys, spec_vals,
                      color=['#e63946','#8338ec','#6c757d','#457b9d'],
                      edgecolor='white', alpha=0.88)
for bar, v in zip(bars2, spec_vals):
    axes2[1].text(v + abs(max(spec_vals))*0.01, bar.get_y() + bar.get_height()/2,
                  f'{v:.4f}', va='center', fontsize=9)
axes2[1].set_xlim(0, max(spec_vals)*1.2)
axes2[1].set_title('Spectral & Path Metrics', fontsize=11, fontweight='bold')
axes2[1].set_xlabel('Value')
axes2[1].grid(True, alpha=0.15, axis='x')

plt.tight_layout()
plt.savefig('./figs/global_metrics.png', dpi=180, bbox_inches='tight')
print("✓ global_metrics.png saved")

# =============================================================================
# FIGURE 3: Centrality correlation heatmap (5 measures)
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(7, 6))
corr = df[centrality_cols].corr()
im   = ax3.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
short_names = ['Degree', 'Betweenness', 'Closeness', 'Eigenvector', 'PageRank']
ax3.set_xticks(range(5)); ax3.set_yticks(range(5))
ax3.set_xticklabels(short_names, fontsize=9, rotation=30, ha='right')
ax3.set_yticklabels(short_names, fontsize=9)
for i in range(5):
    for j in range(5):
        ax3.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=9,
                 color='white' if abs(corr.iloc[i,j]) > 0.6 else 'black')
plt.colorbar(im, ax=ax3, label='Pearson r', shrink=0.8)
ax3.set_title('Centrality Measure Correlation\n(82 regions: cortical + subcortical)',
              fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('./figs/centrality_correlation.png', dpi=180, bbox_inches='tight')
print("✓ centrality_correlation.png saved")

# Save CSVs
df.to_csv('./data/hub_centrality_table.csv', index=False)
pd.DataFrame([global_metrics]).T.rename(columns={0:'Value'})\
  .to_csv('./data/global_metrics.csv')
print("✓ CSVs saved")

# =============================================================================
# CENTRALITY EQUATIONS IN LaTeX FORM
# =============================================================================
LATEX_EQUATIONS = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CENTRALITY MEASURES — LaTeX Equations                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. Degree Centrality                                                        ║
║     C_D(i) = \frac{k_i}{N - 1}                                               ║
║     where  k_i = \sum_{j} A_{ij}  (binary degree of node i),  N = |V|        ║
║                                                                              ║
║  2. Betweenness Centrality                                                   ║
║     C_B(i) = \frac{1}{(N-1)(N-2)}                                            ║
║              \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}       ║
║     where  \sigma_{st}        = total shortest paths from s to t             ║
║            \sigma_{st}(i)     = those that pass through i                   ║
║     (shortest paths use inverse-weight distances: d = 1/w)                  ║
║                                                                              ║
║  3. Closeness Centrality                                                     ║
║     C_C(i) = \frac{N - 1}{\sum_{j \neq i} d(i, j)}                          ║
║     where  d(i,j) = geodesic distance (sum of 1/w along shortest path)      ║
║                                                                              ║
║  4. Eigenvector Centrality                                                   ║
║     x_i = \frac{1}{\lambda_{\max}} \sum_{j} A_{ij}\, x_j                    ║
║     i.e.  \mathbf{A}\,\mathbf{x} = \lambda_{\max}\,\mathbf{x}               ║
║     where  A_{ij} = weighted adjacency,  \lambda_{\max} = spectral radius   ║
║                                                                              ║
║  5. PageRank (Eigenvalue) Centrality  [α = 0.85]                             ║
║     x_i = \alpha \sum_{j} A_{ij}\,\frac{x_j}{k_j^{\text{out}}}             ║
║           + \frac{1 - \alpha}{N}                                             ║
║     where  k_j^{\text{out}} = \sum_k A_{jk}  (weighted out-degree / strength)║
║            \alpha = 0.85  (damping / restart probability)                   ║
║            \beta  = \frac{1-\alpha}{N}  (uniform teleportation term)         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║              GLOBAL CENTRALITY AGGREGATES — LaTeX Equations                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  For any node-level centrality measure C_X(i):                              ║
║                                                                              ║
║     C_X^{\text{global}} = \frac{1}{N} \sum_{i=1}^{N} C_X(i)                ║
║                                                                              ║
║  Applied here:                                                               ║
║     C_D^{\text{global}}   = \frac{1}{N} \sum_i C_D(i)   (Global Degree)    ║
║     C_B^{\text{global}}   = \frac{1}{N} \sum_i C_B(i)   (Global Betweenness)║
║     C_C^{\text{global}}   = \frac{1}{N} \sum_i C_C(i)   (Global Closeness) ║
║     C_{EV}^{\text{global}}= \frac{1}{N} \sum_i C_{EV}(i)(Global Eigenvector)║
║     C_{PR}^{\text{global}}= \frac{1}{N} \sum_i x_i      (Global PageRank)  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║              SPECTRAL / TOPOLOGICAL METRICS — LaTeX Equations               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Density:                                                                    ║
║     \rho = \frac{|E|}{\binom{N}{2}} = \frac{2|E|}{N(N-1)}                   ║
║                                                                              ║
║  Global Efficiency:                                                          ║
║     E_{\text{glob}} = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{1}{d(i,j)}     ║
║                                                                              ║
║  Average Clustering Coefficient (weighted):                                  ║
║     \bar{C} = \frac{1}{N} \sum_i C_i,\quad                                  ║
║     C_i = \frac{1}{k_i(k_i-1)} \sum_{j,h} (w_{ij} w_{ih} w_{jh})^{1/3}    ║
║                                                                              ║
║  Transitivity:                                                               ║
║     T = \frac{3 \times \text{(# triangles)}}{\text{(# connected triples)}}  ║
║                                                                              ║
║  Spectral Radius:                                                            ║
║     \lambda_{\max} = \max_i \lambda_i(\mathbf{A})                            ║
║                                                                              ║
║  Algebraic Connectivity (Fiedler value):                                     ║
║     \mu_2 = \lambda_2(\mathbf{L}),\quad                                      ║
║     \mathbf{L} = \mathbf{D} - \mathbf{A}                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
print(LATEX_EQUATIONS)

print("\nDone.")
