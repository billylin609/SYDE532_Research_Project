"""
HCP Community Detection — combined visualization.

Figure 1: Three-method comparison (sorted adjacency matrices)
Figure 2: Leiden focused — detailed matrix + community breakdown
Figure 3: Leiden brain partition — axial anatomy view
"""
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
import numpy as np
import networkx as nx

from HCP_Community.model import (
    load_full_sc, detect_greedy_modularity, detect_leiden,
    detect_spectral, label_communities, NETWORK_COLORS,
)

_DIR     = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)
np.random.seed(42)

BG           = '#0a0a12'
COMM_PALETTE = ['#E05C5C', '#4A90D9', '#27AE60', '#F39C12',
                '#8E44AD', '#1ABC9C', '#E67E22', '#BDC3C7']

# ── Load data once ─────────────────────────────────────────────────────────────
sc_ctx, sc_ctx_labels, n, A_bin, G_w, G_uw, short_labels, known_networks = load_full_sc()
strength_arr = sc_ctx.sum(axis=1)
btw_arr      = np.array(list(nx.betweenness_centrality(G_uw, normalized=True).values()))
print(f'Loaded {n} regions\n')

# ── Run algorithms ─────────────────────────────────────────────────────────────
labels_gm, Q_gm, _ = detect_greedy_modularity(G_uw)
labels_ld, Q_ld    = detect_leiden(sc_ctx, n)
k                  = len(np.unique(labels_ld))
labels_sp, Q_sp    = detect_spectral(sc_ctx, k, G_uw)

methods = [
    ('Greedy Modularity', labels_gm, Q_gm, '#E05C5C'),
    ('Leiden',            labels_ld, Q_ld, '#4A90D9'),
    ('Spectral',          labels_sp, Q_sp, '#27AE60'),
]

# ── Terminal: centrality arrays for per-community use ─────────────────────────
deg_c = np.array(list(nx.degree_centrality(G_uw).values()))
btw_c = np.array(list(nx.betweenness_centrality(G_uw, normalized=True).values()))
G_close = G_w.copy()
for u, v, d in G_close.edges(data=True):
    d['distance'] = 1.0 / (d['weight'] + 1e-9)
cls_c = np.array(list(nx.closeness_centrality(G_close, distance='distance').values()))

# ── Terminal: modularity Q per algorithm ──────────────────────────────────────
print()
print('=' * 50)
print('MODULARITY Q PER ALGORITHM')
print('=' * 50)
print(f'  {"Algorithm":<25} {"Communities":>13} {"Q":>10}')
print(f'  {"-"*25} {"-"*13} {"-"*10}')
for name, labs, Q, _ in methods:
    print(f'  {name:<25} {len(np.unique(labs)):>13} {Q:>10.4f}')

# ── Terminal: per-community global centrality for Leiden ──────────────────────
print()
print('=' * 72)
print('LEIDEN — PER-COMMUNITY CENTRALITY (from full graph)')
print('=' * 72)
print(f'  {"Community":<28} {"Size":>5}  {"Global Degree":>14} {"Global Betweenness":>19} {"Global Closeness":>17}')
print(f'  {"-"*28} {"-"*5}  {"-"*14} {"-"*19} {"-"*17}')

labelled_ld = label_communities(labels_ld, 'Leiden', sc_ctx_labels,
                                 known_networks, strength_arr, btw_arr)
for cid in np.unique(labels_ld):
    members = np.where(labels_ld == cid)[0]
    fn = labelled_ld.loc[labelled_ld['Community ID'] == cid, 'Functional Label'].values[0]
    label_str = f'C{cid}: {fn.split("(")[0].strip()[:22]}'
    print(f'  {label_str:<28} {len(members):>5}  {deg_c[members].mean():>14.6f} {btw_c[members].mean():>19.6f} {cls_c[members].mean():>17.6f}')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Three-method comparison: sorted adjacency matrices
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=BG)
fig.suptitle('Community Detection — Sorted Adjacency Matrix Comparison',
             color='white', fontsize=15, fontweight='bold', y=1.01)

for ax, (name, labs, Q, accent) in zip(axes, methods):
    ax.set_facecolor('#0f0f1a')

    sort_order = np.argsort(labs)
    sc_sorted  = sc_ctx[np.ix_(sort_order, sort_order)]
    _, counts  = np.unique(labs[sort_order], return_counts=True)
    boundaries = np.cumsum(counts)[:-1] - 0.5
    n_comm     = len(counts)

    im = ax.imshow(np.log1p(sc_sorted), cmap='inferno', aspect='auto',
                   interpolation='nearest')

    for b in boundaries:
        ax.axhline(b, color=accent, lw=1.2, alpha=0.9)
        ax.axvline(b, color=accent, lw=1.2, alpha=0.9)

    labelled = label_communities(labs, name, sc_ctx_labels, known_networks,
                                  strength_arr, btw_arr)
    cum = 0
    for i, cnt in enumerate(counts):
        mid = cum + cnt / 2
        cid = np.unique(labs[sort_order])[i]
        fn  = labelled.loc[labelled['Community ID'] == cid, 'Functional Label'].values[0]
        ax.text(mid, mid, f'C{cid}\n{fn.split("(")[0].strip()[:12]}',
                ha='center', va='center', fontsize=6, color='white', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
        cum += cnt

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='log(streamlines + 1)')
    ax.set_title(f'{name}\nQ = {Q:.4f}  |  {n_comm} communities',
                  color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Region (sorted by community)', color='#aabbcc', fontsize=9)
    ax.set_ylabel('Region (sorted by community)', color='#aabbcc', fontsize=9)
    ax.tick_params(colors='#667788', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334455')

plt.tight_layout(rect=[0, 0, 1, 1])
out1 = os.path.join(FIGS_DIR, 'hcp_community_compare.png')
plt.savefig(out1, dpi=150, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out1}')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Leiden focused: detailed matrix + community breakdown
# ══════════════════════════════════════════════════════════════════════════════
labelled_ld = label_communities(labels_ld, 'Leiden', sc_ctx_labels,
                                 known_networks, strength_arr, btw_arr)

sort_order_ld = np.argsort(labels_ld)
sc_sorted_ld  = sc_ctx[np.ix_(sort_order_ld, sort_order_ld)]
unique_ld, counts_ld = np.unique(labels_ld[sort_order_ld], return_counts=True)
boundaries_ld = np.cumsum(counts_ld)[:-1] - 0.5
n_comm_ld     = len(unique_ld)
comm_colors_ld = {c: COMM_PALETTE[i % len(COMM_PALETTE)]
                  for i, c in enumerate(unique_ld)}

fig, ax = plt.subplots(1, 1, figsize=(9, 8), facecolor=BG)
fig.suptitle(f'Leiden Community Detection — Detailed View  (Q = {Q_ld:.4f})',
             color='white', fontsize=14, fontweight='bold')

ax.set_facecolor('#0f0f1a')
im = ax.imshow(np.log1p(sc_sorted_ld), cmap='inferno', aspect='auto',
               interpolation='nearest')
for b in boundaries_ld:
    ax.axhline(b, color='#4A90D9', lw=1.5, alpha=0.95)
    ax.axvline(b, color='#4A90D9', lw=1.5, alpha=0.95)

# Community midpoint tick labels
mids = []
cum  = 0
for i, (cid, cnt) in enumerate(zip(unique_ld, counts_ld)):
    mid = cum + cnt / 2
    fn  = labelled_ld.loc[labelled_ld['Community ID'] == cid, 'Functional Label'].values[0]
    ax.text(mid, mid, f'C{cid}\n{fn.split("(")[0].strip()[:12]}',
            ha='center', va='center', fontsize=6.5, color='white', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=1.8, foreground='black')])
    # Color band on the left edge
    ax.barh(cum - 0.5, -3, height=cnt, left=0, color=comm_colors_ld[cid],
            align='edge', zorder=5)
    mids.append(mid)
    cum += cnt

plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='log(streamlines + 1)')
ax.set_title('Adjacency Matrix (sorted by community)', color='white',
              fontsize=11, fontweight='bold')
ax.set_xlabel('Region (sorted)', color='#aabbcc', fontsize=9)
ax.set_ylabel('Region (sorted)', color='#aabbcc', fontsize=9)
ax.tick_params(colors='#667788', labelsize=7)
for spine in ax.spines.values():
    spine.set_edgecolor('#334455')

plt.tight_layout()
out2 = os.path.join(FIGS_DIR, 'hcp_leiden_detail.png')
plt.savefig(out2, dpi=150, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out2}')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — All three methods: brain anatomy partition view
# ══════════════════════════════════════════════════════════════════════════════

# MNI centroids for DK-68 cortical atlas (left hemisphere — mirrored for right)
DK_MNI_LEFT = {
    'bankssts':               (-54, -40, 12),
    'caudalanteriorcingulate': ( -8,  12, 38),
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
    'rostralanteriorcingulate':( -6,  38,  6),
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

from enigmatoolbox.datasets import load_sc as _load_sc
_, ctx_lbl, _, sctx_lbl = _load_sc()

coords = []
for lbl in ctx_lbl:
    s        = str(lbl).replace('L_', '').replace('R_', '').lower()
    is_right = str(lbl).startswith('R_')
    xyz      = DK_MNI_LEFT.get(s, (0, 0, 0))
    coords.append((-xyz[0] if is_right else xyz[0], xyz[1], xyz[2]))
for lbl in sctx_lbl:
    coords.append(SCTX_MNI.get(str(lbl).lower(), (0, 0, 0)))
coords = np.array(coords)
xy     = coords[:, :2]   # axial projection (x=LR, y=AP)

is_sctx      = np.array([i >= 68 for i in range(n)])
edges        = list(G_w.edges(data=True))
edge_weights = np.array([d['weight'] for _, _, d in edges])
nonzero_w    = edge_weights[edge_weights > 0]
threshold    = np.percentile(nonzero_w, 55)      # top 45% of actual connections
w_min, w_max = threshold, nonzero_w.max()

KEY = {'precuneus', 'posteriorcingulate', 'precentral', 'superiorfrontal',
       'insula', 'superiortemporal', 'inferiorparietal', 'thal', 'hippo'}

fig, axes = plt.subplots(1, 3, figsize=(27, 9), facecolor=BG)
fig.suptitle('Community Detection — Brain Anatomy View  (Axial MNI Projection)',
             color='white', fontsize=15, fontweight='bold', y=1.01)

for ax, (name, labs, Q, _accent) in zip(axes, methods):
    unique_comms = np.unique(labs)
    comm_color   = {c: COMM_PALETTE[i % len(COMM_PALETTE)]
                    for i, c in enumerate(unique_comms)}
    labelled_m   = label_communities(labs, name, sc_ctx_labels,
                                      known_networks, strength_arr, btw_arr)

    ax.set_facecolor(BG)
    ax.set_aspect('equal')
    ax.axis('off')

    # Brain silhouette
    ax.add_patch(Ellipse((0, -10), width=155, height=182,
                         facecolor='#0c111e', edgecolor='#2a3a4a',
                         linewidth=2.0, zorder=0))
    ax.axvline(0, color='#2a3a4a', lw=1.0, alpha=0.6, zorder=1)
    ax.text(-62, 82, 'L', color='#5a7a8a', fontsize=12, fontweight='bold', ha='center')
    ax.text( 62, 82, 'R', color='#5a7a8a', fontsize=12, fontweight='bold', ha='center')
    ax.text(  0, 96, 'Anterior',  color='#4a6a7a', fontsize=8, ha='center')
    ax.text(  0,-98, 'Posterior', color='#4a6a7a', fontsize=8, ha='center')

    # Top-50% edges, thickness scaled by weight
    for (u, v, d), w in zip(edges, edge_weights):
        if w < threshold or u >= n or v >= n:
            continue
        w_norm = (w - w_min) / (w_max - w_min + 1e-9)
        same   = labs[u] == labs[v]
        col    = comm_color[labs[u]] if same else '#1a2a3a'
        alp    = 0.55 if same else 0.12
        lw     = 0.4 + 2.6 * w_norm if same else 0.3 + 0.7 * w_norm
        ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]],
                '-', color=col, lw=lw, alpha=alp, zorder=2)

    # Nodes
    for i in range(n):
        x, y   = xy[i]
        color  = comm_color[labs[i]]
        sz     = 70 if is_sctx[i] else 40
        marker = 's' if is_sctx[i] else 'o'
        ax.scatter(x, y, s=sz * 4, c=[color], alpha=0.18,
                   edgecolors='none', zorder=3, marker=marker)
        ax.scatter(x, y, s=sz, c=[color],
                   edgecolors='white', linewidths=0.6, zorder=4, marker=marker)

    # Key region labels
    for i, lbl in enumerate(sc_ctx_labels):
        s = str(lbl).replace('L_', '').replace('R_', '').lower()
        if any(k in s for k in KEY):
            ax.text(xy[i, 0], xy[i, 1] + 5.5, s[:10],
                    ha='center', va='bottom', fontsize=5, color='#ccdde8', zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground='black')])

    ax.set_xlim(-95, 95)
    ax.set_ylim(-110, 110)
    ax.set_title(f'{name}\nQ = {Q:.4f}  |  {len(unique_comms)} communities',
                  color='white', fontsize=11, fontweight='bold', pad=10)

    # Per-panel community legend
    handles = []
    for cid in unique_comms:
        fn   = labelled_m.loc[labelled_m['Community ID'] == cid, 'Functional Label'].values[0]
        size = int((labs == cid).sum())
        handles.append(mpatches.Patch(
            color=comm_color[cid],
            label=f'C{cid}: {fn.split("(")[0].strip()[:16]} (n={size})'))
    ax.legend(handles=handles, loc='lower left',
              facecolor='#0d0d1a', edgecolor='#334455',
              labelcolor='white', fontsize=7, framealpha=0.92, borderpad=0.6)

plt.tight_layout()
out3 = os.path.join(FIGS_DIR, 'hcp_brain_partition.png')
plt.savefig(out3, dpi=150, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out3}')
