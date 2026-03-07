"""
HCP Young Adult Structural Connectivity (SC) Consensus Matrix
==============================================================

This script loads and explores the group-consensus structural connectivity
matrix from the Human Connectome Project (HCP) Young Adult dataset, as
provided by the ENIGMA Toolbox.

Key References:
- Betzel et al. (2019). Distance-dependent consensus thresholds for generating
  group-representative structural brain networks. Network Neuroscience, 3(2), 475-496.
- Van Essen et al. (2013). The WU-Minn Human Connectome Project: An overview. NeuroImage, 80, 62-79.
- Larivière et al. (2021). The ENIGMA Toolbox. Nature Methods, 18, 698-700.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# =============================================================================
# 1. Load the SC consensus matrix using ENIGMA Toolbox
# =============================================================================
from enigmatoolbox.datasets import load_sc

# Load cortico-cortical SC (default: Desikan-Killiany 'aparc' with 68 regions)
sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()

print("=" * 70)
print("HCP Young Adult — Structural Connectivity Consensus Matrix")
print("=" * 70)

# --- Basic shape and metadata ---
print(f"\n[1] CORTICO-CORTICAL SC MATRIX")
print(f"    Shape: {sc_ctx.shape}")
print(f"    Number of cortical regions: {sc_ctx.shape[0]}")
print(f"    Data type: {sc_ctx.dtype}")
print(f"    Region labels (first 10): {sc_ctx_labels[:10].tolist()}")

print(f"\n[2] SUBCORTICO-CORTICAL SC MATRIX")
print(f"    Shape: {sc_sctx.shape}")
print(f"    Subcortical labels: {sc_sctx_labels.tolist()}")

# --- Value statistics ---
print(f"\n[3] VALUE STATISTICS (cortico-cortical)")
print(f"    Min:    {sc_ctx.min():.6f}")
print(f"    Max:    {sc_ctx.max():.6f}")
print(f"    Mean:   {sc_ctx.mean():.6f}")
print(f"    Median: {np.median(sc_ctx):.6f}")
print(f"    Std:    {sc_ctx.std():.6f}")

# --- Sparsity ---
total_possible = sc_ctx.shape[0] * (sc_ctx.shape[0] - 1) / 2  # upper triangle
nonzero_edges = np.count_nonzero(np.triu(sc_ctx, k=1))
density = nonzero_edges / total_possible * 100
print(f"\n[4] NETWORK SPARSITY")
print(f"    Total possible edges (upper tri): {int(total_possible)}")
print(f"    Non-zero edges:                   {nonzero_edges}")
print(f"    Density:                          {density:.1f}%")

# --- Symmetry check ---
is_symmetric = np.allclose(sc_ctx, sc_ctx.T)
print(f"\n[5] SYMMETRY CHECK")
print(f"    Is symmetric: {is_symmetric}")

# --- Node degree (number of connections per region) ---
binary_sc = (sc_ctx > 0).astype(int)
np.fill_diagonal(binary_sc, 0)
degree = binary_sc.sum(axis=0)
print(f"\n[6] NODE DEGREE (binary, number of connections)")
print(f"    Mean degree:   {degree.mean():.1f}")
print(f"    Max degree:    {degree.max()} ({sc_ctx_labels[degree.argmax()]})")
print(f"    Min degree:    {degree.min()} ({sc_ctx_labels[degree.argmin()]})")

# --- Node strength (sum of weighted connections per region) ---
strength = sc_ctx.sum(axis=0)
print(f"\n[7] NODE STRENGTH (weighted, sum of connection weights)")
print(f"    Mean strength: {strength.mean():.2f}")
print(f"    Strongest hub: {sc_ctx_labels[strength.argmax()]} ({strength.max():.2f})")
print(f"    Weakest node:  {sc_ctx_labels[strength.argmin()]} ({strength.min():.2f})")

# =============================================================================
# 2. Also load with different parcellations for comparison
# =============================================================================
print(f"\n{'=' * 70}")
print("AVAILABLE PARCELLATIONS")
print("=" * 70)
parcellations = ['aparc', 'schaefer_100', 'schaefer_200', 'schaefer_300', 'schaefer_400', 'glasser_360']
for parc in parcellations:
    try:
        sc_tmp, labels_tmp, _, _ = load_sc(parcellation=parc)
        print(f"  {parc:>15s}:  {sc_tmp.shape[0]:>4d} regions  |  "
              f"density = {np.count_nonzero(np.triu(sc_tmp, k=1)) / (sc_tmp.shape[0]*(sc_tmp.shape[0]-1)/2)*100:.1f}%")
    except Exception as e:
        print(f"  {parc:>15s}:  Error — {e}")

# =============================================================================
# 3. Visualization
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("HCP Young Adult — Structural Connectivity Consensus Matrix\n"
             "(Desikan-Killiany 68-region parcellation)", fontsize=14, fontweight='bold')

# --- (a) Full connectivity matrix ---
ax = axes[0, 0]
# Use log scale for better visibility since weights span orders of magnitude
sc_plot = sc_ctx.copy()
sc_plot[sc_plot == 0] = np.nan
im = ax.imshow(sc_plot, cmap='Blues', aspect='equal',
               norm=LogNorm(vmin=np.nanmin(sc_plot[sc_plot > 0]), vmax=np.nanmax(sc_plot)))
ax.set_title("(a) Weighted SC Matrix (log scale)", fontsize=11)
ax.set_xlabel("Brain Region Index")
ax.set_ylabel("Brain Region Index")
# Add hemisphere divider
n_regions = sc_ctx.shape[0]
ax.axhline(y=n_regions/2 - 0.5, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
ax.axvline(x=n_regions/2 - 0.5, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
ax.text(n_regions/4, -2, 'LH', ha='center', fontsize=9, color='red')
ax.text(3*n_regions/4, -2, 'RH', ha='center', fontsize=9, color='red')
plt.colorbar(im, ax=ax, label='Connection Weight', shrink=0.8)

# --- (b) Binary adjacency (thresholded) ---
ax = axes[0, 1]
ax.imshow(binary_sc, cmap='Greys', aspect='equal')
ax.set_title(f"(b) Binary Adjacency (density = {density:.1f}%)", fontsize=11)
ax.set_xlabel("Brain Region Index")
ax.set_ylabel("Brain Region Index")
ax.axhline(y=n_regions/2 - 0.5, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
ax.axvline(x=n_regions/2 - 0.5, color='red', linewidth=0.8, linestyle='--', alpha=0.7)

# --- (c) Edge weight distribution ---
ax = axes[1, 0]
upper_tri = sc_ctx[np.triu_indices_from(sc_ctx, k=1)]
nonzero_weights = upper_tri[upper_tri > 0]
ax.hist(nonzero_weights, bins=80, color='steelblue', edgecolor='white', linewidth=0.3)
ax.set_title("(c) Distribution of Non-Zero Edge Weights", fontsize=11)
ax.set_xlabel("Connection Weight (streamline count)")
ax.set_ylabel("Number of Edges")
ax.set_yscale('log')
ax.text(0.95, 0.95, f"N edges = {len(nonzero_weights)}\nMedian = {np.median(nonzero_weights):.2f}",
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# --- (d) Node strength bar chart (top 20) ---
ax = axes[1, 1]
sorted_idx = np.argsort(strength)[::-1][:20]
colors = ['#2171b5' if 'L_' in sc_ctx_labels[i] or 'lh_' in sc_ctx_labels[i].lower()
          else '#cb181d' for i in sorted_idx]
bars = ax.barh(range(20), strength[sorted_idx], color=colors, edgecolor='white', linewidth=0.3)
ax.set_yticks(range(20))
ax.set_yticklabels([sc_ctx_labels[i] for i in sorted_idx], fontsize=7)
ax.invert_yaxis()
ax.set_title("(d) Top 20 Regions by Node Strength", fontsize=11)
ax.set_xlabel("Total Connection Strength")

plt.tight_layout()
plt.savefig('./figs/hcp_sc_overview.png', dpi=180, bbox_inches='tight')
print(f"\n✓ Figure saved to hcp_sc_overview.png")

# =============================================================================
# 4. Subcortico-cortical connectivity snapshot
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(14, 4))
im2 = ax2.imshow(sc_sctx, cmap='YlOrRd', aspect='auto')
ax2.set_yticks(range(len(sc_sctx_labels)))
ax2.set_yticklabels(sc_sctx_labels, fontsize=7)
ax2.set_xlabel("Cortical Region Index (Desikan-Killiany)")
ax2.set_title("Subcortico-Cortical Structural Connectivity", fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Connection Weight', shrink=0.8)
plt.tight_layout()
plt.savefig('./figs/hcp_sc_subcortical.png', dpi=180, bbox_inches='tight')
print(f"✓ Subcortical figure saved to hcp_sc_subcortical.png")

print(f"\nDone! All labels:\n{sc_ctx_labels.tolist()}")