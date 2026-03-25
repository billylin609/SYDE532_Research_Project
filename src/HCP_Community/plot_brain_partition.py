"""Plot community partitions mapped onto 2D brain anatomy (axial MNI projection)."""
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
import numpy as np
import networkx as nx

from HCP_Community.model import (
    load_full_sc, detect_greedy_modularity, detect_leiden,
    detect_spectral, label_communities, NETWORK_COLORS,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)
np.random.seed(42)

# ── MNI centroid coordinates for DK atlas (68 cortical) + 14 subcortical ──────
# Order matches ENIGMA load_sc() label order: 34 L + 34 R cortical, then 14 sctx
# Coordinates: (x=LR, y=AP, z=SI) — we project axially using (x, y)
DK_MNI_LEFT = {
    'bankssts':               (-54, -40, 12),
    'caudalanteriorcingulate':( -8,  12, 38),
    'caudalmiddlefrontal':    (-36,  14, 46),
    'cuneus':                 (-12, -82, 26),
    'entorhinal':             (-24,  -8,-28),
    'fusiform':               (-36, -40,-18),
    'inferiorparietal':       (-46, -56, 32),
    'inferiortemporal':       (-52, -28,-18),
    'isthmuscingulate':       ( -8, -38, 26),
    'lateraloccipital':       (-30, -82, 14),
    'lateralorbitofrontal':   (-24,  34,-14),
    'lingual':                (-14, -66, -4),
    'medialorbitofrontal':    ( -8,  50,-12),
    'middletemporal':         (-58, -18,-10),
    'parahippocampal':        (-22, -28,-18),
    'paracentral':            ( -8, -24, 66),
    'parsopercularis':        (-50,  16, 14),
    'parsorbitalis':          (-32,  34, -8),
    'parstriangularis':       (-44,  30,  8),
    'pericalcarine':          (-10, -76, 10),
    'postcentral':            (-40, -32, 58),
    'posteriorcingulate':     ( -8, -46, 30),
    'precentral':             (-40,  -8, 58),
    'precuneus':              ( -8, -64, 42),
    'rostralanteriorcingulate':(-6,  38,  6),
    'rostralmiddlefrontal':   (-28,  42, 24),
    'superiorfrontal':        (-14,  36, 52),
    'superiorparietal':       (-22, -60, 60),
    'superiortemporal':       (-54, -14,  4),
    'supramarginal':          (-56, -42, 32),
    'frontalpole':            (-10,  62,  4),
    'temporalpole':           (-38,  14,-28),
    'transversetemporal':     (-44, -24, 10),
    'insula':                 (-34,   0,  8),
}

SCTX_MNI = {
    'laccumb': (-10,  12, -8), 'raccumb': ( 10,  12, -8),
    'lamyg':   (-22,  -4,-18), 'ramyg':   ( 22,  -4,-18),
    'lcaud':   (-14,  10, 10), 'rcaud':   ( 14,  10, 10),
    'lhippo':  (-26, -20,-14), 'rhippo':  ( 26, -20,-14),
    'lpal':    (-20,   0, -2), 'rpal':    ( 20,   0, -2),
    'lput':    (-26,   6,  0), 'rput':    ( 26,   6,  0),
    'lthal':   (-14, -18,  8), 'rthal':   ( 14, -18,  8),
}


def get_mni_coords(sc_ctx_labels, sc_sctx_labels=None):
    """Return (x, y, z) MNI arrays for the 82-region label set."""
    coords = []
    for lbl in sc_ctx_labels:
        s = str(lbl).replace('L_', '').replace('R_', '').lower()
        is_right = str(lbl).startswith('R_')
        xyz = DK_MNI_LEFT.get(s, (0, 0, 0))
        x = -xyz[0] if is_right else xyz[0]
        coords.append((x, xyz[1], xyz[2]))

    if sc_sctx_labels is not None:
        for lbl in sc_sctx_labels:
            s = str(lbl).lower()
            coords.append(SCTX_MNI.get(s, (0, 0, 0)))

    return np.array(coords)


# ── Load data ──────────────────────────────────────────────────────────────────
sc_ctx, sc_ctx_labels, n, A_bin, G_w, G_uw, short_labels, known_networks = load_full_sc()
from enigmatoolbox.datasets import load_sc as _load_sc
_, ctx_lbl, _, sctx_lbl = _load_sc()

strength_arr = sc_ctx.sum(axis=1)
btw_arr = np.array(list(nx.betweenness_centrality(G_uw, normalized=True).values()))

coords   = get_mni_coords(ctx_lbl, sctx_lbl)
xy       = coords[:, :2]   # axial projection: x=LR, y=AP

# ── Community detection ────────────────────────────────────────────────────────
labels_gm, Q_gm, _ = detect_greedy_modularity(G_uw)
labels_ld, Q_ld    = detect_leiden(sc_ctx, n)
k = len(np.unique(labels_ld))
labels_sp, Q_sp    = detect_spectral(sc_ctx, k, G_uw)

methods = [
    ('Greedy Modularity', labels_gm, Q_gm),
    ('Leiden',            labels_ld, Q_ld),
    ('Spectral',          labels_sp, Q_sp),
]

COMM_PALETTE = ['#E05C5C', '#4A90D9', '#27AE60',
                '#F39C12', '#8E44AD', '#1ABC9C']

BG = '#0a0a12'

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 8), facecolor=BG)
fig.suptitle('Community Detection — Brain Anatomy View  (Axial MNI Projection)',
             color='white', fontsize=16, fontweight='bold', y=1.01)

for ax, (name, labs, Q) in zip(axes, methods):
    ax.set_facecolor('#0a0a12')
    ax.set_aspect('equal')
    ax.axis('off')

    unique_comms = np.unique(labs)
    n_comm = len(unique_comms)
    comm_color = {c: COMM_PALETTE[i % len(COMM_PALETTE)]
                  for i, c in enumerate(unique_comms)}

    # ── Brain outline (axial silhouette) ──────────────────────────────────────
    brain = Ellipse((0, -10), width=150, height=175,
                    facecolor='#0f1520', edgecolor='#334455',
                    linewidth=1.5, zorder=0)
    ax.add_patch(brain)
    # Midline
    ax.axvline(0, color='#334455', lw=0.8, alpha=0.5, zorder=1)
    # Hemisphere labels
    ax.text(-55, 80, 'L', color='#445566', fontsize=12, fontweight='bold', ha='center')
    ax.text( 55, 80, 'R', color='#445566', fontsize=12, fontweight='bold', ha='center')
    ax.text(  0, 90, 'Anterior', color='#445566', fontsize=8, ha='center')
    ax.text(  0,-95, 'Posterior', color='#445566', fontsize=8, ha='center')

    # ── Edges — draw top-weight edges only to avoid clutter ──────────────────
    edges       = list(G_w.edges(data=True))
    edge_weights= np.array([d['weight'] for _, _, d in edges])
    threshold   = np.percentile(edge_weights, 92)   # top 8% edges

    for u, v, d in edges:
        if d['weight'] < threshold:
            continue
        same_comm = labs[u] == labs[v]
        x0, y0 = xy[u]; x1, y1 = xy[v]
        col = comm_color[labs[u]] if same_comm else '#223344'
        alp = 0.45 if same_comm else 0.12
        lw  = 0.8  if same_comm else 0.4
        ax.plot([x0, x1], [y0, y1], '-', color=col,
                lw=lw, alpha=alp, zorder=2)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    is_sctx = np.array([i >= 68 for i in range(n)])

    for i in range(n):
        x, y  = xy[i]
        color = comm_color[labs[i]]
        sz    = 55 if is_sctx[i] else 35
        marker= 's' if is_sctx[i] else 'o'
        # Glow
        ax.scatter(x, y, s=sz*3.5, c=[color], alpha=0.15,
                   edgecolors='none', zorder=3, marker=marker)
        # Node
        ax.scatter(x, y, s=sz, c=[color],
                   edgecolors='white', linewidths=0.5,
                   zorder=4, marker=marker)

    # ── Labels for hub/key regions only ───────────────────────────────────────
    key_regions = {
        'precuneus', 'posteriorcingulate', 'precentral', 'superiorfrontal',
        'hippocampus', 'insula', 'superiortemporal', 'inferiorparietal',
        'thal', 'hippo', 'amyg',
    }
    for i, lbl in enumerate(sc_ctx_labels):
        s = str(lbl).replace('L_','').replace('R_','').lower()
        if any(k in s for k in key_regions):
            ax.text(xy[i, 0], xy[i, 1] + 5, s[:9],
                    ha='center', va='bottom', fontsize=5,
                    color='#ccddee', zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground='black')])

    ax.set_xlim(-90, 90)
    ax.set_ylim(-105, 105)
    ax.set_title(f'{name}\nQ = {Q:.4f}  |  {n_comm} communities',
                  color='white', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
out = os.path.join(FIGS_DIR, 'hcp_brain_partition.png')
plt.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
