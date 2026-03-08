"""
HCP Young Adult Structural Connectivity (SC) Consensus Matrix
==============================================================

This script loads and explores the group-consensus structural connectivity
matrix from the Human Connectome Project (HCP) Young Adult dataset, as
provided by the ENIGMA Toolbox.

Full 82×82 matrix includes:
  - Cortico-cortical SC        (68×68, upper-left block)
  - Subcortico-cortical SC     (14×68, lower-left / upper-right blocks)
  - Subcortico-subcortical SC  (14×14, lower-right block — zeros, not in load_sc())

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

sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()

n_ctx  = sc_ctx.shape[0]   # 68 cortical regions
n_sctx = sc_sctx.shape[0]  # 14 subcortical regions
N      = n_ctx + n_sctx    # 82 total regions

# Build full 82×82 SC matrix
# [ ctx-ctx  (68×68) | ctx-sctx  (68×14) ]
# [ sctx-ctx (14×68) | sctx-sctx (14×14) <- zeros, not available in load_sc() ]
sc_full = np.zeros((N, N))
sc_full[:n_ctx, :n_ctx] = sc_ctx           # cortico-cortical
sc_full[:n_ctx, n_ctx:] = sc_sctx.T        # cortical-subcortical
sc_full[n_ctx:, :n_ctx] = sc_sctx          # subcortical-cortical
# sc_full[n_ctx:, n_ctx:] remains zero (subcortico-subcortical not in load_sc)

labels = list(sc_ctx_labels) + list(sc_sctx_labels)

print("=" * 70)
print("HCP Young Adult — Structural Connectivity Consensus Matrix (82×82)")
print("=" * 70)

# --- Basic shape and metadata ---
print(f"\n[1] CORTICO-CORTICAL SC MATRIX")
print(f"    Shape: {sc_ctx.shape}")
print(f"    Number of cortical regions: {n_ctx}")
print(f"    Data type: {sc_ctx.dtype}")
print(f"    Region labels (first 10): {sc_ctx_labels[:10].tolist()}")

print(f"\n[2] SUBCORTICO-CORTICAL SC MATRIX")
print(f"    Shape: {sc_sctx.shape}")
print(f"    Subcortical labels: {sc_sctx_labels.tolist()}")

print(f"\n[3] FULL SC MATRIX (82×82)")
print(f"    Shape: {sc_full.shape}")
print(f"    Cortical regions:    {n_ctx}")
print(f"    Subcortical regions: {n_sctx}")
print(f"    Total regions:       {N}")

# --- Value statistics (per block) ---
def block_stats(mat, name):
    upper = mat[np.triu_indices_from(mat, k=1)]
    nz = upper[upper > 0]
    total_pairs = len(upper)
    print(f"\n[4] VALUE STATISTICS — {name}")
    if len(nz) == 0:
        print(f"    No non-zero edges.")
        return
    print(f"    Min (non-zero): {nz.min():.6f}")
    print(f"    Max:            {nz.max():.6f}")
    print(f"    Mean (non-zero):{nz.mean():.6f}")
    print(f"    Density:        {len(nz)/total_pairs*100:.1f}%  ({len(nz)}/{total_pairs} pairs)")

block_stats(sc_full[:n_ctx, :n_ctx], "Cortico-cortical (68×68)")
# For subcortico-cortical, the full rectangular block
sctx_vals = sc_sctx[sc_sctx > 0]
total_sctx_pairs = n_sctx * n_ctx
print(f"\n[5] VALUE STATISTICS — Subcortico-cortical (14×68)")
if len(sctx_vals) > 0:
    print(f"    Min (non-zero): {sctx_vals.min():.6f}")
    print(f"    Max:            {sctx_vals.max():.6f}")
    print(f"    Mean (non-zero):{sctx_vals.mean():.6f}")
    print(f"    Density:        {(sc_sctx > 0).sum()/total_sctx_pairs*100:.1f}%  ({(sc_sctx > 0).sum()}/{total_sctx_pairs} pairs)")
else:
    print(f"    No non-zero edges.")
block_stats(sc_full[n_ctx:, n_ctx:], "Subcortico-subcortical (14×14, zeros)")

# --- Sparsity for full matrix ---
total_possible = N * (N - 1) / 2
nonzero_edges = np.count_nonzero(np.triu(sc_full, k=1))
density = nonzero_edges / total_possible * 100
print(f"\n[6] FULL MATRIX NETWORK SPARSITY")
print(f"    Total possible edges (upper tri): {int(total_possible)}")
print(f"    Non-zero edges:                   {nonzero_edges}")
print(f"    Density:                          {density:.1f}%")

# --- Symmetry check ---
is_symmetric = np.allclose(sc_full, sc_full.T)
print(f"\n[7] SYMMETRY CHECK (full 82×82)")
print(f"    Is symmetric: {is_symmetric}")

# --- Node degree ---
binary_sc = (sc_full > 0).astype(int)
np.fill_diagonal(binary_sc, 0)
degree = binary_sc.sum(axis=0)
print(f"\n[8] NODE DEGREE (binary, all 82 regions)")
print(f"    Mean degree:   {degree.mean():.1f}")
print(f"    Max degree:    {degree.max()} ({labels[degree.argmax()]})")
print(f"    Min degree:    {degree.min()} ({labels[degree.argmin()]})")
print(f"    Cortical mean:    {degree[:n_ctx].mean():.1f}")
print(f"    Subcortical mean: {degree[n_ctx:].mean():.1f}")

# --- Node strength ---
strength = sc_full.sum(axis=0)
print(f"\n[9] NODE STRENGTH (weighted, all 82 regions)")
print(f"    Mean strength: {strength.mean():.2f}")
print(f"    Strongest hub: {labels[strength.argmax()]} ({strength.max():.2f})")
print(f"    Weakest node:  {labels[strength.argmin()]} ({strength.min():.2f})")

# =============================================================================
# 2. Also load with different parcellations for comparison
# =============================================================================
print(f"\n{'=' * 70}")
print("AVAILABLE PARCELLATIONS")
print("=" * 70)
parcellations = ['aparc', 'schaefer_100', 'schaefer_200', 'schaefer_300', 'schaefer_400', 'glasser_360']
for parc in parcellations:
    try:
        sc_tmp, labels_tmp, sc_sctx_tmp, _ = load_sc(parcellation=parc)
        n_c = sc_tmp.shape[0]
        n_s = sc_sctx_tmp.shape[0]
        n_total = n_c + n_s
        density_c = np.count_nonzero(np.triu(sc_tmp, k=1)) / (n_c*(n_c-1)/2)*100
        density_s = (sc_sctx_tmp > 0).sum() / (n_s*n_c)*100
        print(f"  {parc:>15s}:  ctx={n_c}, sctx={n_s}, total={n_total}  |  "
              f"ctx density={density_c:.1f}%  sctx density={density_s:.1f}%")
    except Exception as e:
        print(f"  {parc:>15s}:  Error — {e}")

# =============================================================================
# 3. Visualization
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle("HCP Young Adult — Structural Connectivity Consensus Matrix (82×82)\n"
             "(Desikan-Killiany 68 cortical + 14 subcortical regions)", fontsize=14, fontweight='bold')

# --- (a) Full 82×82 connectivity matrix ---
ax = axes[0, 0]
sc_plot = sc_full.copy()
sc_plot[sc_plot == 0] = np.nan
nz_vals = sc_plot[~np.isnan(sc_plot)]
im = ax.imshow(sc_plot, cmap='Blues', aspect='equal',
               norm=LogNorm(vmin=nz_vals.min(), vmax=nz_vals.max()))
ax.set_title("(a) Full 82×82 Weighted SC Matrix (log scale)", fontsize=10)
ax.set_xlabel("Brain Region Index")
ax.set_ylabel("Brain Region Index")
# Hemisphere divider within cortex
ax.axhline(y=n_ctx/2 - 0.5, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
ax.axvline(x=n_ctx/2 - 0.5, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
# Cortex/subcortex divider
ax.axhline(y=n_ctx - 0.5, color='white', linewidth=1.5, linestyle='-')
ax.axvline(x=n_ctx - 0.5, color='white', linewidth=1.5, linestyle='-')
ax.text(n_ctx/4, -2.5, 'LH ctx', ha='center', fontsize=8, color='red')
ax.text(3*n_ctx/4, -2.5, 'RH ctx', ha='center', fontsize=8, color='red')
ax.text(n_ctx + n_sctx/2, -2.5, 'sctx', ha='center', fontsize=8, color='white')
plt.colorbar(im, ax=ax, label='Connection Weight', shrink=0.8)

# --- (b) Binary adjacency ---
ax = axes[0, 1]
ax.imshow(binary_sc, cmap='Greys', aspect='equal')
ax.set_title(f"(b) Binary Adjacency (density = {density:.1f}%)", fontsize=10)
ax.set_xlabel("Brain Region Index")
ax.set_ylabel("Brain Region Index")
ax.axhline(y=n_ctx/2 - 0.5, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
ax.axvline(x=n_ctx/2 - 0.5, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
ax.axhline(y=n_ctx - 0.5, color='red', linewidth=1.5, linestyle='-')
ax.axvline(x=n_ctx - 0.5, color='red', linewidth=1.5, linestyle='-')

# --- (c) Edge weight distribution by block ---
ax = axes[0, 2]
ctx_weights = sc_ctx[np.triu_indices_from(sc_ctx, k=1)]
ctx_nz = ctx_weights[ctx_weights > 0]
sctx_nz = sc_sctx[sc_sctx > 0]
ax.hist(ctx_nz, bins=60, color='steelblue', edgecolor='white', linewidth=0.3,
        alpha=0.75, label=f'Cortico-cortical (n={len(ctx_nz)})')
ax.hist(sctx_nz, bins=40, color='darkorange', edgecolor='white', linewidth=0.3,
        alpha=0.75, label=f'Subcortico-cortical (n={len(sctx_nz)})')
ax.set_title("(c) Edge Weight Distribution by Block", fontsize=10)
ax.set_xlabel("Connection Weight")
ax.set_ylabel("Number of Edges")
ax.set_yscale('log')
ax.legend(fontsize=8)

# --- (d) Node degree: cortical vs subcortical ---
ax = axes[1, 0]
ctx_colors = ['#2171b5'] * n_ctx
sctx_colors = ['#d95f02'] * n_sctx
all_colors = ctx_colors + sctx_colors
sorted_idx = np.argsort(degree)[::-1][:30]
bar_colors = [all_colors[i] for i in sorted_idx]
ax.barh(range(30), degree[sorted_idx], color=bar_colors, edgecolor='white', linewidth=0.3)
ax.set_yticks(range(30))
ax.set_yticklabels([labels[i] for i in sorted_idx], fontsize=6)
ax.invert_yaxis()
ax.set_title("(d) Top 30 Regions by Degree\n(blue=cortical, orange=subcortical)", fontsize=10)
ax.set_xlabel("Degree (# connections)")

# --- (e) Node strength bar chart (top 30) ---
ax = axes[1, 1]
sorted_str = np.argsort(strength)[::-1][:30]
str_colors = [all_colors[i] for i in sorted_str]
ax.barh(range(30), strength[sorted_str], color=str_colors, edgecolor='white', linewidth=0.3)
ax.set_yticks(range(30))
ax.set_yticklabels([labels[i] for i in sorted_str], fontsize=6)
ax.invert_yaxis()
ax.set_title("(e) Top 30 Regions by Node Strength\n(blue=cortical, orange=subcortical)", fontsize=10)
ax.set_xlabel("Total Connection Strength")

# --- (f) Subcortico-cortical connectivity snapshot ---
ax = axes[1, 2]
im2 = ax.imshow(sc_sctx, cmap='YlOrRd', aspect='auto')
ax.set_yticks(range(n_sctx))
ax.set_yticklabels(sc_sctx_labels, fontsize=7)
ax.set_xlabel("Cortical Region Index (Desikan-Killiany)")
ax.set_title("(f) Subcortico-Cortical SC Block (14×68)", fontsize=10)
plt.colorbar(im2, ax=ax, label='Connection Weight', shrink=0.8)

plt.tight_layout()
plt.savefig('./figs/hcp_sc_overview.png', dpi=180, bbox_inches='tight')
print(f"\n✓ Figure saved to hcp_sc_overview.png")

# =============================================================================
# 4. Per-block summary figure
# =============================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("HCP SC — Block Structure of Full 82×82 Matrix", fontsize=13, fontweight='bold')

block_labels = ['Cortico-cortical\n(68×68)', 'Subcortico-cortical\n(14×68)', 'Subcortico-subcortical\n(14×14, zeros)']
block_mats   = [sc_full[:n_ctx, :n_ctx], sc_sctx, sc_full[n_ctx:, n_ctx:]]
block_cmaps  = ['Blues', 'YlOrRd', 'Greys']

for ax, mat, title, cmap in zip(axes2, block_mats, block_labels, block_cmaps):
    nz = mat[mat > 0]
    if len(nz) > 0:
        disp = mat.copy(); disp[disp == 0] = np.nan
        im3 = ax.imshow(disp, cmap=cmap, aspect='auto',
                        norm=LogNorm(vmin=nz.min(), vmax=nz.max()))
        plt.colorbar(im3, ax=ax, label='Weight', shrink=0.8)
    else:
        ax.imshow(mat, cmap=cmap, aspect='auto')
        ax.text(0.5, 0.5, 'No data\n(zeros)', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    ax.set_title(title, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('./figs/hcp_sc_blocks.png', dpi=180, bbox_inches='tight')
print(f"✓ Block structure figure saved to hcp_sc_blocks.png")

print(f"\nDone! Labels ({N} total):")
print(f"  Cortical ({n_ctx}):    {sc_ctx_labels.tolist()}")
print(f"  Subcortical ({n_sctx}): {sc_sctx_labels.tolist()}")
