"""Plot only the Leiden sorted adjacency matrix."""
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from HCP_Community.model import load_sc_with_networks, detect_leiden

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

sc_ctx, sc_ctx_labels, n, A_bin, G_w, G_uw, short_labels, known_networks = load_sc_with_networks()
labels_leiden, Q_leiden = detect_leiden(sc_ctx, n)

sort_order = np.argsort(labels_leiden)
sc_sorted  = sc_ctx[np.ix_(sort_order, sort_order)]

_, counts    = np.unique(labels_leiden[sort_order], return_counts=True)
boundaries   = np.cumsum(counts)[:-1] - 0.5

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(np.log1p(sc_sorted), cmap='inferno', aspect='auto')
for b in boundaries:
    ax.axhline(b, color='cyan', lw=1.5)
    ax.axvline(b, color='cyan', lw=1.5)
plt.colorbar(im, ax=ax, label='log(streamlines + 1)')
ax.set_title('Adjacency Matrix — Leiden Communities\n(sorted by community)',
              fontsize=12, fontweight='bold')
ax.set_xlabel('Region (sorted)')
ax.set_ylabel('Region (sorted)')

plt.tight_layout()
out = os.path.join(FIGS_DIR, 'hcp_leiden_matrix_only.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
