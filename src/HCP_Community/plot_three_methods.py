"""Compare Greedy, Leiden, and Spectral community detection — terminal + matrix plot."""
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx

from HCP_Community.model import (
    load_full_sc,
    detect_greedy_modularity,
    detect_leiden,
    detect_spectral,
    label_communities,
    NETWORK_COLORS,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
sc_ctx, sc_ctx_labels, n, A_bin, G_w, G_uw, short_labels, known_networks = load_full_sc()
strength_arr = sc_ctx.sum(axis=1)
btw_arr      = np.array(list(nx.betweenness_centrality(G_uw, normalized=True).values()))
print(f'Loaded {n} regions (68 cortical + 14 subcortical)\n')

# ── Run three algorithms ───────────────────────────────────────────────────────
labels_gm,  Q_gm,  _  = detect_greedy_modularity(G_uw)
labels_ld,  Q_ld       = detect_leiden(sc_ctx, n)
k = len(np.unique(labels_ld))          # match Leiden's k for fair comparison
labels_sp,  Q_sp       = detect_spectral(sc_ctx, k, G_uw)

methods = [
    ('Greedy Modularity', labels_gm, Q_gm, '#E05C5C'),
    ('Leiden',            labels_ld, Q_ld, '#4A90D9'),
    ('Spectral',          labels_sp, Q_sp, '#27AE60'),
]

# ── Terminal output ────────────────────────────────────────────────────────────
print('=' * 62)
print(f'{"Algorithm":<22} {"Communities":>13} {"Modularity Q":>13}')
print('-' * 62)
for name, labs, Q, _ in methods:
    n_comm = len(np.unique(labs))
    print(f'{name:<22} {n_comm:>13} {Q:>13.4f}')
print('=' * 62)

for name, labs, Q, _ in methods:
    labelled = label_communities(labs, name, sc_ctx_labels, known_networks,
                                  strength_arr, btw_arr)
    print(f'\n── {name}  (Q={Q:.4f}, {len(np.unique(labs))} communities) ──')
    print(labelled[['Community ID', 'Size', 'Functional Label', 'Purity']].to_string(index=False))

# ── Plot: sorted adjacency matrix for each method ─────────────────────────────
BG = '#0a0a12'
fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=BG)
fig.suptitle('Community Detection — Sorted Adjacency Matrix',
             color='white', fontsize=16, fontweight='bold', y=1.01)

for ax, (name, labs, Q, accent) in zip(axes, methods):
    ax.set_facecolor('#0f0f1a')

    # Sort nodes by community
    sort_order = np.argsort(labs)
    sc_sorted  = sc_ctx[np.ix_(sort_order, sort_order)]

    # Community boundaries
    _, counts    = np.unique(labs[sort_order], return_counts=True)
    boundaries   = np.cumsum(counts)[:-1] - 0.5
    n_comm       = len(counts)

    # Matrix
    im = ax.imshow(np.log1p(sc_sorted), cmap='inferno', aspect='auto',
                    interpolation='nearest')

    # Community boundary lines
    for b in boundaries:
        ax.axhline(b, color=accent, lw=1.2, alpha=0.9)
        ax.axvline(b, color=accent, lw=1.2, alpha=0.9)

    # Community labels on diagonal
    cum = 0
    labelled = label_communities(labs, name, sc_ctx_labels, known_networks,
                                  strength_arr, btw_arr)
    for i, cnt in enumerate(counts):
        mid = cum + cnt / 2
        cid = np.unique(labs[sort_order])[i]
        fn  = labelled.loc[labelled['Community ID'] == cid, 'Functional Label'].values[0]
        short = fn.split('(')[0].strip()[:14]
        ax.text(mid, mid, f'C{cid}\n{short}',
                ha='center', va='center', fontsize=6,
                color='white', fontweight='bold',
                path_effects=[__import__('matplotlib.patheffects', fromlist=['withStroke'])
                               .withStroke(linewidth=1.5, foreground='black')])
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

# Shared legend
handles = [mpatches.Patch(color=c, label=net)
           for net, c in NETWORK_COLORS.items() if net != 'Unassigned']
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
           facecolor='#0f0f1a', edgecolor='#334455',
           labelcolor='white', framealpha=0.95,
           bbox_to_anchor=(0.5, -0.06))

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = os.path.join(FIGS_DIR, 'hcp_three_methods.png')
plt.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# ── Plot: partition graph for each method ─────────────────────────────────────
import matplotlib.patheffects as pe

# Compute layout once — reuse for all three panels
print('Computing graph layout…')
pos = nx.spring_layout(G_uw, seed=42, k=1.8 / np.sqrt(n))

# Edge weights for alpha scaling
weights = np.array([G_w[u][v]['weight'] for u, v in G_uw.edges()])
w_norm  = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)

# Palette: up to 8 distinct community colours
COMM_PALETTE = ['#E05C5C', '#4A90D9', '#27AE60', '#F39C12',
                '#8E44AD', '#1ABC9C', '#E67E22', '#BDC3C7']

fig2, axes2 = plt.subplots(1, 3, figsize=(21, 8), facecolor=BG)
fig2.suptitle('Community Detection — Partition Graph',
              color='white', fontsize=16, fontweight='bold', y=1.01)

for ax, (name, labs, Q, accent) in zip(axes2, methods):
    ax.set_facecolor('#0f0f1a')
    ax.axis('off')

    unique_comms = np.unique(labs)
    comm_color   = {c: COMM_PALETTE[i % len(COMM_PALETTE)]
                    for i, c in enumerate(unique_comms)}
    node_colors  = [comm_color[labs[n]] for n in range(len(labs))]

    # Draw edges — thin, low alpha, coloured by weight
    edges = list(G_uw.edges())
    edge_colors = [plt.cm.Blues(0.3 + 0.6 * w_norm[i]) for i, _ in enumerate(edges)]
    nx.draw_networkx_edges(G_uw, pos, ax=ax, edge_color=edge_colors,
                           width=0.5, alpha=0.35)

    # Draw nodes
    nx.draw_networkx_nodes(G_uw, pos, ax=ax,
                           node_color=node_colors,
                           node_size=120,
                           edgecolors='white', linewidths=0.6)

    # Node labels — only show region short name, small font
    nx.draw_networkx_labels(G_uw, pos, ax=ax,
                            labels={i: short_labels[i][:6] for i in range(n)},
                            font_size=4.5, font_color='white')

    # Community convex-hull shading
    for cid in unique_comms:
        members = [i for i in range(n) if labs[i] == cid]
        if len(members) < 3:
            continue
        pts = np.array([pos[m] for m in members])
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            poly = plt.Polygon(hull_pts, closed=True,
                               facecolor=comm_color[cid], alpha=0.10,
                               edgecolor=comm_color[cid], linewidth=1.2,
                               linestyle='--')
            ax.add_patch(poly)
        except Exception:
            pass

    # Community legend patches
    labelled = label_communities(labs, name, sc_ctx_labels, known_networks,
                                  strength_arr, btw_arr)
    handles = []
    for cid in unique_comms:
        fn   = labelled.loc[labelled['Community ID'] == cid, 'Functional Label'].values[0]
        size = int((labs == cid).sum())
        short_fn = fn.split('(')[0].strip()[:18]
        handles.append(mpatches.Patch(color=comm_color[cid],
                                       label=f'C{cid}: {short_fn} (n={size})'))
    ax.legend(handles=handles, loc='lower left',
              facecolor='#0d0d1a', edgecolor='#334455',
              labelcolor='white', fontsize=7.5, framealpha=0.92,
              borderpad=0.6)

    ax.set_title(f'{name}\nQ = {Q:.4f}  |  {len(unique_comms)} communities',
                  color='white', fontsize=11, fontweight='bold', pad=8)

plt.tight_layout()
out2 = os.path.join(FIGS_DIR, 'hcp_three_methods_partition.png')
plt.savefig(out2, dpi=150, facecolor=BG, bbox_inches='tight')
plt.close(fig2)
print(f'Saved: {out2}')
