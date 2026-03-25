# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.10.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # HCP Brain Network Centrality Analysis

# %%
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

from HCP_Centrality.model import run_all

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

# %matplotlib inline
print('Imports successful.')

# %% — Run full pipeline
(sc_ctx, sc_ctx_labels, n_ctx, G,
 degree_arr, betweenness_arr, closeness_arr, eigenvec_arr,
 eigenvalues_sorted, lambda1, v1, centrality_df) = run_all()

# %% — Visualize: top-15 bar charts for each centrality measure
measures = [
    ('Degree',      degree_arr,      'steelblue'),
    ('Betweenness', betweenness_arr, 'darkorange'),
    ('Closeness',   closeness_arr,   'mediumseagreen'),
    ('Eigenvector', eigenvec_arr,    'mediumpurple'),
]

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for ax, (name, arr, color) in zip(axes, measures):
    top15_idx = np.argsort(arr)[::-1][:15]
    labels_top = [sc_ctx_labels[i] for i in top15_idx]
    values_top = arr[top15_idx]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, 15))[::-1] if color == 'steelblue' else \
             plt.cm.Oranges(np.linspace(0.4, 0.9, 15))[::-1] if color == 'darkorange' else \
             plt.cm.Greens(np.linspace(0.4, 0.9, 15))[::-1] if color == 'mediumseagreen' else \
             plt.cm.Purples(np.linspace(0.4, 0.9, 15))[::-1]

    ax.barh(labels_top[::-1], values_top[::-1], color=colors)
    ax.set_title(f'{name} Centrality — Top 15 Regions', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'{name} centrality score')
    ax.tick_params(axis='y', labelsize=9)

plt.suptitle('HCP SC Network — Centrality Measures (Cortical Regions)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_centrality_bars.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Eigenvalue spectrum
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(eigenvalues_sorted, 'o-', color='steelblue', markersize=4)
axes[0].axhline(0, color='k', linewidth=0.8, linestyle='--')
axes[0].axvline(0, color='tomato', linewidth=1.5, linestyle='--', label=f'λ₁ = {lambda1:.2f}')
axes[0].set_title('Eigenvalue Spectrum of SC Matrix', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Eigenvalue index (descending)')
axes[0].set_ylabel('Eigenvalue')
axes[0].legend()

v1_abs = np.abs(v1)
sorted_v1_idx = np.argsort(v1_abs)[::-1]
axes[1].bar(range(n_ctx), v1_abs[sorted_v1_idx], color='mediumpurple', edgecolor='white', linewidth=0.3)
axes[1].set_title('Principal Eigenvector |v₁| (region loadings)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Region (sorted by |v₁|)')
axes[1].set_ylabel('|v₁| loading')
for rank, i in enumerate(sorted_v1_idx[:5]):
    axes[1].text(rank, v1_abs[i] + 0.001, sc_ctx_labels[i], fontsize=7,
                 ha='center', va='bottom', rotation=45)

plt.suptitle(f'Spectral Analysis  |  λ₁ = {lambda1:.4f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_eigenspectrum.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Cross-measure scatter matrix
measure_names = ['Degree', 'Betweenness', 'Closeness', 'Eigenvector']
measure_arrays = [degree_arr, betweenness_arr, closeness_arr, eigenvec_arr]
n_m = len(measure_names)

fig, axes = plt.subplots(n_m, n_m, figsize=(14, 14))

for i in range(n_m):
    for j in range(n_m):
        ax = axes[i, j]
        if i == j:
            ax.hist(measure_arrays[i], bins=20, color='slategray', edgecolor='white')
            ax.set_title(measure_names[i], fontsize=10, fontweight='bold')
        else:
            r = np.corrcoef(measure_arrays[j], measure_arrays[i])[0, 1]
            ax.scatter(measure_arrays[j], measure_arrays[i], s=18, alpha=0.6,
                       color='steelblue', edgecolors='none')
            ax.set_title(f'r = {r:.2f}', fontsize=9)
        if i == n_m - 1:
            ax.set_xlabel(measure_names[j], fontsize=9)
        if j == 0:
            ax.set_ylabel(measure_names[i], fontsize=9)

plt.suptitle('Centrality Cross-Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_centrality_scatter.png'), dpi=150, bbox_inches='tight')
plt.show()
