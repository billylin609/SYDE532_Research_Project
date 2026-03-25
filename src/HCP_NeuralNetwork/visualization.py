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
# # HCP Brain Network — Topology, Roles & Efficiency–Resilience Tradeoffs

# %%
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from HCP_NeuralNetwork.model import (
    load_sc_graphs, compute_weighted_vs_unweighted, detect_communities,
    classify_node_roles, compute_topological_metrics, compute_weighted_resilience,
    robustness_curve,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

# %matplotlib inline
print('Imports successful.')

# %% — Load data
sc_ctx, sc_ctx_labels, n, A_bin, G_w, G_uw = load_sc_graphs()
print(f'Nodes: {n}, Weighted edges: {G_w.number_of_edges()}, Connected: {__import__("networkx").is_connected(G_w)}')

# %% — Section 1: Weighted vs Unweighted Comparison
deg_w, deg_uw, btw_w, btw_uw, eig_w, eig_uw, r_deg, r_btw, r_eig = compute_weighted_vs_unweighted(
    sc_ctx, sc_ctx_labels, G_w, G_uw, n
)

print('Spearman rank correlation (weighted vs unweighted):')
print(f'  Degree centrality    : r = {r_deg:.3f}')
print(f'  Betweenness centrality: r = {r_btw:.3f}')
print(f'  Eigenvector centrality: r = {r_eig:.3f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

pairs = [
    (deg_uw, deg_w, 'Degree', r_deg, 'steelblue'),
    (btw_uw, btw_w, 'Betweenness', r_btw, 'darkorange'),
    (eig_uw, eig_w, 'Eigenvector', r_eig, 'mediumpurple'),
]

for ax, (x, y, name, r, color) in zip(axes, pairs):
    ax.scatter(x, y, c=color, s=50, alpha=0.7, edgecolors='k', linewidths=0.3)
    diff = np.abs(stats.zscore(x) - stats.zscore(y))
    for i in np.argsort(diff)[::-1][:5]:
        ax.annotate(sc_ctx_labels[i], (x[i], y[i]), fontsize=7,
                    xytext=(4, 2), textcoords='offset points')
    ax.set_title(f'{name} Centrality\nSpearman r = {r:.3f}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Unweighted')
    ax.set_ylabel('Weighted')

plt.suptitle('Weighted vs Unweighted Centrality Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_weighted_vs_unweighted.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 2: Node Role Classification
communities, community_arr = detect_communities(G_uw, n, sc_ctx_labels)
role_df = classify_node_roles(deg_w, btw_w, eig_w, deg_uw, sc_ctx_labels, community_arr, n)

print('Node role counts:')
print(role_df['Role'].value_counts().to_string())
print()
for role in ['Hub', 'Broker', 'Bridge']:
    regions = role_df[role_df['Role'] == role]['Region'].tolist()
    print(f'{role:10s}: {regions}')

role_colors = {'Hub': 'tomato', 'Broker': 'darkorange', 'Bridge': 'steelblue', 'Peripheral': 'lightgray'}
role_sizes = {'Hub': 200, 'Broker': 120, 'Bridge': 100, 'Peripheral': 40}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for role, grp in role_df.groupby('Role'):
    axes[0].scatter(grp['Strength'], grp['Betweenness'],
                    c=role_colors[role], s=role_sizes[role],
                    label=role, edgecolors='k', linewidths=0.4, alpha=0.85, zorder=3)
for _, row in role_df[role_df['Role'] != 'Peripheral'].iterrows():
    axes[0].annotate(row['Region'], (row['Strength'], row['Betweenness']),
                     fontsize=7, xytext=(4, 2), textcoords='offset points')

axes[0].set_xlabel('Weighted Strength', fontsize=12)
axes[0].set_ylabel('Betweenness Centrality', fontsize=12)
axes[0].set_title('Node Role Map\n(Strength vs Betweenness)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)

counts = role_df['Role'].value_counts()
bars = axes[1].bar(counts.index, counts.values,
                   color=[role_colors[r] for r in counts.index],
                   edgecolor='k', linewidth=0.6)
axes[1].bar_label(bars, fontsize=11)
axes[1].set_title('Node Role Distribution', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Number of regions')

plt.suptitle('Brain Network Node Roles — HCP SC (Cortical)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_node_roles.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 3: Efficiency–Resilience Tradeoff
d_hat, lambda1_uw, tau, hub_node, lambda1_prime, Lambda, alpha, beta = compute_topological_metrics(
    G_uw, A_bin, deg_uw, n
)

print('── Topological Metrics ──────────────────────────')
print(f'  Average path length d̂     : {d_hat:.4f}')
print(f'  Spectral radius λ₁         : {lambda1_uw:.4f}')
print(f'  Epidemic threshold τ        : {tau:.4f}')
print(f'  Hub removed                : {sc_ctx_labels[hub_node]}')
print(f'  λ₁ after removal           : {lambda1_prime:.4f}')
print(f'  Node-removal resilience Λ  : {Lambda:.2f}%')
print()
print('── Tradeoff Parameters ──────────────────────────')
print(f'  α (node attack)  : {alpha:.4f}  ({"efficient" if alpha > 0.5 else "resilient"}-dominant)')
print(f'  β (epidemic)     : {beta:.4f}  ({"efficient" if beta > 0.5 else "resilient"}-dominant)')

W_out, R_gamma, E_gamma, E_norm, R_norm, rho, xi = compute_weighted_resilience(sc_ctx, G_uw, n)

print('── Weighted Metrics ─────────────────────────────')
print(f'  Weighted resilience R(Γ)   : {R_gamma:.4f}')
print(f'  Weighted efficiency E(Γ)   : {E_gamma:.6f}')
print(f'  Ẽ (normalised efficiency)  : {E_norm:.4f}')
print()
print('── Cooperation Parameters ───────────────────────')
print(f'  ρ = {rho:.4f}  → {"efficiency" if rho > 1 else "resilience"}-dominated')
print(f'  ξ = {xi:.4f}  → {"well" if xi > 1 else "poorly"}-designed ({xi:.2f}/2.0)')

# %% — Robustness curves
order_w = np.argsort(W_out)[::-1]
order_uw = np.argsort(deg_uw)[::-1]

lcc_w, apl_w, lam_w = robustness_curve(sc_ctx, order_w)
lcc_uw, apl_uw, lam_uw = robustness_curve(A_bin, order_uw)

steps_w = np.arange(1, len(lcc_w) + 1) / n
steps_uw = np.arange(1, len(lcc_uw) + 1) / n

R_idx_w = np.trapz(lcc_w, steps_w)
R_idx_uw = np.trapz(lcc_uw, steps_uw)
print(f'Robustness index (weighted) : {R_idx_w:.4f}')
print(f'Robustness index (binary)   : {R_idx_uw:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(steps_w, lcc_w, 'o-', color='tomato', markersize=3, label=f'Weighted  (R={R_idx_w:.3f})')
axes[0].plot(steps_uw, lcc_uw, 's-', color='steelblue', markersize=3, label=f'Unweighted (R={R_idx_uw:.3f})')
axes[0].set_title('Robustness Curve\n(LCC size after targeted hub removal)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Fraction of nodes removed')
axes[0].set_ylabel('Largest Connected Component (fraction)')
axes[0].legend()
axes[0].axhline(0.5, color='k', linewidth=0.8, linestyle='--', label='50% threshold')

axes[1].plot(steps_w, lam_w, 'o-', color='tomato', markersize=3, label='Weighted')
axes[1].plot(steps_uw, lam_uw, 's-', color='steelblue', markersize=3, label='Unweighted')
axes[1].set_title('Spectral Radius λ₁ during Hub Removal\n(λ₁↓ = epidemic risk reduced)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Fraction of nodes removed')
axes[1].set_ylabel('Spectral radius λ₁')
axes[1].legend()

plt.suptitle('Efficiency–Resilience Tradeoff under Targeted Attack', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_robustness_curve.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Summary dashboard
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

params = [alpha, beta, 1 - R_gamma]
labels_p = [f'α = {alpha:.3f}\n(node attack)', f'β = {beta:.3f}\n(epidemic)', f'1-R = {1-R_gamma:.3f}\n(weighted attack)']
colors_p = ['tomato', 'darkorange', 'steelblue']
bars = axes[0].barh(labels_p, params, color=colors_p, edgecolor='k', linewidth=0.5)
axes[0].axvline(0.5, color='k', linewidth=1.2, linestyle='--', label='Balanced (0.5)')
axes[0].set_xlim(0, 1)
axes[0].set_title('Efficiency–Resilience\nTradeoff Parameters', fontsize=13, fontweight='bold')
axes[0].set_xlabel('→ 0 resilient   |   efficient → 1')
axes[0].legend(fontsize=9)
for bar, val in zip(bars, params):
    axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=10)

role_comm = role_df.groupby(['Community', 'Role']).size().unstack(fill_value=0)
role_order = ['Hub', 'Broker', 'Bridge', 'Peripheral']
role_comm = role_comm.reindex(columns=[r for r in role_order if r in role_comm.columns])
role_comm.plot(kind='bar', ax=axes[1], color=[role_colors[r] for r in role_comm.columns],
               edgecolor='k', linewidth=0.4)
axes[1].set_title('Node Role Composition\nby Community', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Community')
axes[1].set_ylabel('Number of regions')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(fontsize=9)

coop_names = ['ρ (cooperation ratio)', 'ξ (design quality, /2)']
coop_vals = [min(rho, 2.0), xi / 2.0]
cbars = axes[2].bar(coop_names, coop_vals, color=['mediumpurple', 'mediumseagreen'],
                    edgecolor='k', linewidth=0.5)
axes[2].axhline(1.0, color='k', linewidth=1.2, linestyle='--', label='Balanced')
axes[2].set_title('Weighted Cooperation\nParameters', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Value (normalised)')
axes[2].set_ylim(0, max(coop_vals) * 1.2)
axes[2].legend(fontsize=9)
for bar, val, raw in zip(cbars, coop_vals, [rho, xi]):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{raw:.3f}', ha='center', fontsize=11)

plt.suptitle('HCP Brain Network — Efficiency–Resilience Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_efficiency_resilience_summary.png'), dpi=150, bbox_inches='tight')
plt.show()

print('\n── Final Summary ────────────────────────────────────────────────────────')
print(f'  Hubs    : {list(role_df[role_df["Role"]=="Hub"]["Region"][:5])}')
print(f'  Brokers : {list(role_df[role_df["Role"]=="Broker"]["Region"][:5])}')
print(f'  Bridges : {list(role_df[role_df["Role"]=="Bridge"]["Region"][:5])}')
print(f'  α (node attack tradeoff) = {alpha:.4f}  → {"efficiency" if alpha > 0.5 else "resilience"}-dominant')
print(f'  β (epidemic tradeoff)    = {beta:.4f}  → {"efficiency" if beta > 0.5 else "resilience"}-dominant')
print(f'  ρ (cooperation ratio)    = {rho:.4f}  → {"efficiency" if rho > 1 else "resilience"}-dominated')
print(f'  ξ (design quality)       = {xi:.4f} / 2.0')
