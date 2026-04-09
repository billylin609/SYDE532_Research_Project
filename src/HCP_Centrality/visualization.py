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
from scipy.stats import gaussian_kde

from HCP_Centrality.model import run_all

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

# %matplotlib inline
print('Imports successful.')

# %% — Run full pipeline
(sc_ctx, sc_ctx_labels, n_ctx, G,
 degree_arr, betweenness_arr, closeness_arr, eigenvec_arr,
 broker_arr, global_metrics, _, lambda1, _, centrality_df) = run_all()

# ── shared measure list ────────────────────────────────────────────────────────
measures = [
    ('Degree',      degree_arr,      plt.cm.Blues),
    ('Betweenness', betweenness_arr, plt.cm.Oranges),
    ('Closeness',   closeness_arr,   plt.cm.Greens),
    ('Eigenvector', eigenvec_arr,    plt.cm.Purples),
]

# %% — Figure 1: Local centrality (ranked bars + density, 2 rows × 4 cols)
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Row 0 — top-15 ranked bar charts
for ax, (name, arr, cmap) in zip(axes[0], measures):
    top15_idx = np.argsort(arr)[::-1][:15]
    labels_top = [sc_ctx_labels[i] for i in top15_idx]
    values_top = arr[top15_idx]
    colors = cmap(np.linspace(0.4, 0.85, 15))[::-1]

    ax.barh(range(15), values_top[::-1], color=colors)
    ax.set_yticks(range(15))
    ax.set_yticklabels([f'{i+1}. {l}' for i, l in enumerate(labels_top[::-1])], fontsize=8)
    ax.set_title(f'{name} Centrality\nTop 15 Regions', fontsize=11, fontweight='bold')
    ax.set_xlabel('Centrality score', fontsize=9)
    ax.invert_yaxis()

# Row 1 — normalized density distributions
for ax, (name, arr, cmap) in zip(axes[1], measures):
    color = cmap(0.65)
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
    kde = gaussian_kde(arr_norm, bw_method='scott')
    x = np.linspace(0, 1, 300)
    y = kde(x)

    ax.fill_between(x, y, alpha=0.35, color=color)
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_title(f'{name} — Density Distribution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Normalized centrality score', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_xlim(0, 1)

plt.suptitle('HCP SC Network — Local Centrality', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_local_centrality.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Figure 2: Broker identification
def _norm(x):
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)

bet_norm  = _norm(betweenness_arr)
deg_norm  = _norm(degree_arr)
is_broker = centrality_df['is_broker'].values
top15_broker_idx = np.argsort(broker_arr)[::-1][:15]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left — betweenness vs degree scatter
ax = axes[0]
ax.scatter(deg_norm[~is_broker], bet_norm[~is_broker],
           s=25, alpha=0.5, color='steelblue', label='Non-broker')
ax.scatter(deg_norm[is_broker], bet_norm[is_broker],
           s=60, alpha=0.9, color='tomato', edgecolors='darkred', linewidth=0.8, label='Broker')
for i in top15_broker_idx[:8]:
    ax.annotate(sc_ctx_labels[i], (deg_norm[i], bet_norm[i]),
                fontsize=7, xytext=(4, 4), textcoords='offset points')
ax.axline((0.5, 0.5), slope=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Normalized Degree Centrality', fontsize=10)
ax.set_ylabel('Normalized Betweenness Centrality', fontsize=10)
ax.set_title('Broker Identification\n(above diagonal = high betweenness relative to degree)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

# Right — top-15 broker regions ranked
ax = axes[1]
labels_b = [sc_ctx_labels[i] for i in top15_broker_idx]
values_b  = broker_arr[top15_broker_idx]
colors_b  = plt.cm.Reds(np.linspace(0.4, 0.85, 15))[::-1]
ax.barh(range(15), values_b[::-1], color=colors_b)
ax.set_yticks(range(15))
ax.set_yticklabels([f'{i+1}. {l}' for i, l in enumerate(labels_b[::-1])], fontsize=8)
ax.set_title('Top 15 Broker Regions\n(betweenness − degree, normalized)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Broker score', fontsize=9)
ax.invert_yaxis()

plt.suptitle('HCP SC Network — Broker Nodes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_brokers.png'), dpi=150, bbox_inches='tight')
plt.show()
