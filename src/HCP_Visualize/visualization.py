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
# # HCP Dataset Visualization

# %%
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import Image, display

from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical, plot_subcortical

from HCP_Visualize.model import (
    load_sc_data, load_fc_data, compute_weight_stats, compute_hub_df,
    compute_hemisphere_symmetry, compute_rich_club, compute_sc_fc_coupling,
    compute_degree_and_fc_strength,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

# %matplotlib inline
print('Imports successful.')

# %% — Load data
sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels, sc_combined, combined_labels = load_sc_data()
n_ctx = sc_ctx.shape[0]
n_sctx = sc_sctx.shape[0]

print('=== Cortical–Cortical SC (68 × 68) ===')
print('  Shape      :', sc_ctx.shape)
print('  Non-zero   :', np.count_nonzero(sc_ctx))
print('  Min / Max  :', round(sc_ctx.min(), 4), '/', round(sc_ctx.max(), 4))
print('  Labels (first 5):', sc_ctx_labels[:5])

print('\n=== Subcortical–Cortical SC (14 × 68) ===')
print('  Shape      :', sc_sctx.shape)
print('  Non-zero   :', np.count_nonzero(sc_sctx))
print('  Min / Max  :', round(sc_sctx.min(), 4), '/', round(sc_sctx.max(), 4))
print('  Subcortical labels:', sc_sctx_labels)

print('\n=== Combined SC (82 × 82) ===')
print('  Shape      :', sc_combined.shape)
print('  Non-zero   :', np.count_nonzero(sc_combined))
print('  Min / Max  :', round(sc_combined.min(), 4), '/', round(sc_combined.max(), 4))
print('  Total regions:', len(combined_labels))

# %% — Section 1: SC matrix visualization
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

matrices = [
    (np.log1p(sc_ctx),      'Cortical–Cortical SC\n(68 × 68, log-scale)',         'inferno'),
    (np.log1p(sc_sctx),     'Subcortical–Cortical SC\n(14 × 68, log-scale)',       'inferno'),
    (np.log1p(sc_combined), 'Combined SC (ctx + sctx)\n(82 × 82, log-scale)',      'inferno'),
]

for ax, (mat, title, cmap) in zip(axes, matrices):
    im = ax.imshow(mat, cmap=cmap, aspect='auto')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Region index')
    ax.set_ylabel('Region index')
    plt.colorbar(im, ax=ax, label='log(streamlines + 1)', fraction=0.046, pad=0.04)

axes[2].axhline(n_ctx - 0.5, color='cyan', linewidth=1.2, linestyle='--')
axes[2].axvline(n_ctx - 0.5, color='cyan', linewidth=1.2, linestyle='--')
axes[2].text(n_ctx + 1, 2, 'Subcortical', color='cyan', fontsize=9)
axes[2].text(2, 2, 'Cortical', color='cyan', fontsize=9)
axes[1].set_ylabel('Subcortical region index')
axes[1].set_xlabel('Cortical region index')

plt.suptitle('HCP Structural Connectivity Matrices', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_sc_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Feature 1: Weight distribution & sparsity
density, nonzero_weights = compute_weight_stats(sc_ctx)

print(f'Network density     : {density:.3f}  ({len(nonzero_weights)} / {int(sc_ctx.shape[0]*(sc_ctx.shape[0]-1)/2)} edges present)')
print(f'Weight range        : [{nonzero_weights.min():.2f}, {nonzero_weights.max():.2f}]')
print(f'Mean weight         : {nonzero_weights.mean():.2f}')
print(f'Median weight       : {np.median(nonzero_weights):.2f}')
print(f'Std weight          : {nonzero_weights.std():.2f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(nonzero_weights, bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('SC Weight Distribution (raw)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Streamline count')
axes[0].set_ylabel('Number of edges')

axes[1].hist(np.log10(nonzero_weights), bins=50, color='darkorange', edgecolor='white')
axes[1].set_title('SC Weight Distribution (log₁₀)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('log₁₀(streamline count)')
axes[1].set_ylabel('Number of edges')

plt.suptitle('Feature 1 — Weight Structure (Heavy-Tailed)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% — Feature 2: Hub regions
hub_df, sc_strength, sc_bin_degree = compute_hub_df(sc_ctx, sc_ctx_labels)

print('Top 10 hub regions by SC strength:')
print(hub_df.head(10).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

top15 = hub_df.head(15)
colors = plt.cm.plasma(np.linspace(0.2, 0.9, 15))
axes[0].barh(top15['Region'][::-1], top15['Strength'][::-1], color=colors[::-1])
axes[0].set_title('Top 15 Hub Regions (Weighted Strength)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Total streamline strength')

axes[1].scatter(sc_bin_degree, sc_strength, c=sc_strength, cmap='plasma', s=60, edgecolors='k', linewidths=0.4)
for i, label in enumerate(sc_ctx_labels):
    if sc_strength[i] > np.percentile(sc_strength, 90):
        axes[1].annotate(label, (sc_bin_degree[i], sc_strength[i]),
                         fontsize=7, xytext=(4, 2), textcoords='offset points')
axes[1].set_title('Node Degree vs. Weighted Strength', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Binary degree')
axes[1].set_ylabel('Weighted strength')

plt.suptitle('Feature 2 — Hub Regions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% — Feature 3: Hemisphere symmetry
lh_strength, rh_strength, lh_labels, corr_lr, asymmetry = compute_hemisphere_symmetry(
    sc_ctx, sc_ctx_labels, sc_strength
)

print(f'Left–Right strength correlation: r = {corr_lr:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(lh_strength, rh_strength, c='mediumseagreen', s=60, edgecolors='k', linewidths=0.4)
for i, lbl in enumerate(lh_labels):
    if abs(lh_strength[i] - rh_strength[i]) > np.std(lh_strength - rh_strength) * 1.5:
        axes[0].annotate(lbl, (lh_strength[i], rh_strength[i]), fontsize=7,
                         xytext=(4, 2), textcoords='offset points')
lims = [min(lh_strength.min(), rh_strength.min()), max(lh_strength.max(), rh_strength.max())]
axes[0].plot(lims, lims, 'k--', linewidth=1, label='Perfect symmetry')
axes[0].set_title(f'Left vs Right Hemisphere Strength\n(r = {corr_lr:.3f})', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Left hemisphere strength')
axes[0].set_ylabel('Right hemisphere strength')
axes[0].legend()

sorted_idx = np.argsort(asymmetry)
axes[1].barh(np.array(lh_labels)[sorted_idx], asymmetry[sorted_idx],
             color=['tomato' if a > 0 else 'steelblue' for a in asymmetry[sorted_idx]])
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_title('Hemisphere Asymmetry Index\n(positive = LH stronger)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Asymmetry index')
axes[1].tick_params(axis='y', labelsize=7)

plt.suptitle('Feature 3 — Bilateral Hemisphere Symmetry', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% — Feature 4: Rich-club organization
k_levels, rc_coeff, rc_null, rc_norm = compute_rich_club(sc_ctx)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_levels, rc_coeff, 'o-', color='steelblue', markersize=4, label='SC network')
axes[0].fill_between(k_levels,
                     np.nanpercentile(rc_null, 5, axis=0),
                     np.nanpercentile(rc_null, 95, axis=0),
                     alpha=0.3, color='gray', label='Null (5–95%)')
axes[0].set_title('Rich-Club Coefficient φ(k)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Degree threshold k')
axes[0].set_ylabel('φ(k)')
axes[0].legend()

axes[1].plot(k_levels, rc_norm, 'o-', color='darkorange', markersize=4)
axes[1].axhline(1.0, color='k', linestyle='--', linewidth=1, label='No rich-club (= 1)')
axes[1].set_title('Normalized Rich-Club Coefficient φ_norm(k)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Degree threshold k')
axes[1].set_ylabel('φ_norm(k)')
axes[1].legend()

plt.suptitle('Feature 4 — Rich-Club Organization\n(hubs connect disproportionately to other hubs)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# %% — Feature 5: Structure–Function coupling
fc_ctx, fc_ctx_labels, fc_sctx, fc_sctx_labels = load_fc_data()

sc_edges, fc_edges, r, p, sc_node_strength, fc_node_strength, r_node, p_node = compute_sc_fc_coupling(
    sc_ctx, fc_ctx, sc_ctx_labels
)

print(f'SC–FC edge correlation: r = {r:.4f}, p = {p:.2e}')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(np.log1p(sc_edges), fc_edges, alpha=0.15, s=8, color='steelblue')
m, b = np.polyfit(np.log1p(sc_edges), fc_edges, 1)
x_line = np.linspace(np.log1p(sc_edges).min(), np.log1p(sc_edges).max(), 100)
axes[0].plot(x_line, m * x_line + b, 'r-', linewidth=2, label=f'r = {r:.3f}, p = {p:.1e}')
axes[0].set_title('SC–FC Edge Coupling', fontsize=13, fontweight='bold')
axes[0].set_xlabel('log(SC streamline count + 1)')
axes[0].set_ylabel('FC Pearson r')
axes[0].legend()

axes[1].scatter(sc_node_strength, fc_node_strength, c=sc_node_strength,
                cmap='plasma', s=60, edgecolors='k', linewidths=0.4)
for i, lbl in enumerate(sc_ctx_labels):
    if sc_node_strength[i] > np.percentile(sc_node_strength, 92):
        axes[1].annotate(lbl, (sc_node_strength[i], fc_node_strength[i]),
                         fontsize=7, xytext=(4, 2), textcoords='offset points')
m2, b2 = np.polyfit(sc_node_strength, fc_node_strength, 1)
x2 = np.linspace(sc_node_strength.min(), sc_node_strength.max(), 100)
axes[1].plot(x2, m2 * x2 + b2, 'r-', linewidth=2, label=f'r = {r_node:.3f}')
axes[1].set_title('Node SC Strength vs FC Mean Strength', fontsize=13, fontweight='bold')
axes[1].set_xlabel('SC total strength')
axes[1].set_ylabel('FC mean strength')
axes[1].legend()

plt.suptitle('Feature 5 — Structure–Function Coupling\n(stronger SC → stronger FC)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# %% — Node-degree and FC strength distributions
sc_degree, fc_strength = compute_degree_and_fc_strength(sc_ctx, fc_ctx)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(sc_degree, bins=20, color='steelblue', edgecolor='white')
axes[0].set_title('SC Node Degree Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Degree')
axes[0].set_ylabel('Count')

axes[1].hist(fc_strength, bins=20, color='tomato', edgecolor='white')
axes[1].set_title('FC Mean Strength Distribution', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Mean Pearson r')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_distributions.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 2: Cortical Surface Visualization
os.environ['VTK_DEFAULT_RENDER_WINDOW_TYPE'] = 'OSMesa'

sc_degree_arr = (sc_ctx > 0).sum(axis=1)
sc_degree_fsa5 = parcel_to_surface(sc_degree_arr, 'aparc_fsa5')

plot_cortical(
    array_name=sc_degree_fsa5,
    surface_name='fsa5',
    size=(800, 400),
    cmap='viridis',
    color_bar=True,
    color_range=(int(sc_degree_arr.min()), int(sc_degree_arr.max())),
    screenshot=True,
    filename='cortical_sc_degree.png',
)
display(Image(os.path.join(FIGS_DIR, 'cortical_sc_degree.png')))

# %%
fc_strength_fsa5 = parcel_to_surface(fc_strength, 'aparc_fsa5')

plot_cortical(
    array_name=fc_strength_fsa5,
    surface_name='fsa5',
    size=(800, 400),
    cmap='RdBu_r',
    color_bar=True,
    color_range=(-0.3, 0.3),
    screenshot=True,
    filename='cortical_fc_strength.png',
)
display(Image(os.path.join(FIGS_DIR, 'cortical_fc_strength.png')))

# %% — Section 3: Subcortical Surface Visualization
sctx_fc_strength = fc_sctx.mean(axis=1)
print('Subcortical FC strength:', dict(zip(fc_sctx_labels, sctx_fc_strength.round(3))))

plot_subcortical(
    array_name=sctx_fc_strength,
    ventricles=False,
    size=(800, 400),
    cmap='RdBu_r',
    color_bar=True,
    color_range=(-0.3, 0.3),
    screenshot=True,
    filename='subcortical_fc_strength.png',
)
display(Image(os.path.join(FIGS_DIR, 'subcortical_fc_strength.png')))

# %%
sctx_sc_strength = sc_sctx.mean(axis=1)
print('Subcortical SC strength:', dict(zip(sc_sctx_labels, sctx_sc_strength.round(3))))

plot_subcortical(
    array_name=sctx_sc_strength,
    ventricles=False,
    size=(800, 400),
    cmap='viridis',
    color_bar=True,
    color_range=(sctx_sc_strength.min(), sctx_sc_strength.max()),
    screenshot=True,
    filename='subcortical_sc_strength.png',
)
display(Image(os.path.join(FIGS_DIR, 'subcortical_sc_strength.png')))
