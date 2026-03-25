"""Plot the adjacency matrix and network graph side by side."""
import os, sys
os.environ['VTK_DEFAULT_RENDER_WINDOW_TYPE'] = 'OSMesa'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import networkx as nx

from example_motter_lai.model import EDGES, ROLES, POS, N, ROLE_COLORS, ALPHA

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

BG    = '#0a0a12'
PANEL = '#0f0f1a'

G = nx.Graph()
G.add_nodes_from(range(N))
G.add_edges_from(EDGES)

A   = nx.to_numpy_array(G, dtype=int)
L0  = nx.betweenness_centrality(G, normalized=False)
C   = {n: (1 + ALPHA) * L0[n] for n in range(N)}
deg = dict(G.degree())

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 9), facecolor=BG)
gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], left=0.07, right=0.97,
                        top=0.88, bottom=0.12, wspace=0.12)
ax_mat = fig.add_subplot(gs[0])
ax_net = fig.add_subplot(gs[1])

for ax in (ax_mat, ax_net):
    ax.set_facecolor(PANEL)

fig.suptitle('Unweighted Network', color='white', fontsize=20,
             fontweight='bold', y=0.97)

# ── Adjacency Matrix ──────────────────────────────────────────────────────────
img = np.zeros((N, N, 4))
for i in range(N):
    for j in range(N):
        if A[i, j] == 1:
            rgba = mcolors.to_rgba(ROLE_COLORS[ROLES[i]])
            img[i, j] = (min(rgba[0]*1.15, 1), min(rgba[1]*1.15, 1),
                          min(rgba[2]*1.15, 1), 1.0)
        else:
            img[i, j] = mcolors.to_rgba(PANEL)

ax_mat.imshow(img, aspect='equal', interpolation='nearest')
ax_mat.set_title('Adjacency Matrix', color='#7ec8f0', fontsize=13,
                  fontweight='bold', pad=14)

# Grid lines
for k in range(N + 1):
    ax_mat.axhline(k - 0.5, color='#1e2a38', lw=0.8)
    ax_mat.axvline(k - 0.5, color='#1e2a38', lw=0.8)

# "1" annotations
for i in range(N):
    for j in range(N):
        if A[i, j] == 1:
            ax_mat.text(j, i, '1', ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold')

# Tick labels — node number + role on y-axis, just number on x-axis
ax_mat.set_xticks(range(N))
ax_mat.set_yticks(range(N))
ax_mat.set_xticklabels([str(n) for n in range(N)], fontsize=10)
ax_mat.set_yticklabels(
    [f'{n}  {ROLES[n]}' for n in range(N)], fontsize=9)
for i, tick in enumerate(ax_mat.get_xticklabels()):
    tick.set_color(ROLE_COLORS[ROLES[i]])
for i, tick in enumerate(ax_mat.get_yticklabels()):
    tick.set_color(ROLE_COLORS[ROLES[i]])

ax_mat.set_xlabel('Node  j', color='#7ec8f0', fontsize=11, labelpad=10)
ax_mat.set_ylabel('Node  i', color='#7ec8f0', fontsize=11, labelpad=10)
ax_mat.tick_params(axis='both', which='both', length=0, pad=6)
for spine in ax_mat.spines.values():
    spine.set_edgecolor('#1e2a38')

# ── Network Graph ─────────────────────────────────────────────────────────────
ax_net.set_aspect('equal')
ax_net.axis('off')
ax_net.set_title('Network Topology', color='#7ec8f0', fontsize=13,
                  fontweight='bold', pad=14)

# Expand axes limits to give room for labels
xs = [POS[n][0] for n in range(N)]
ys = [POS[n][1] for n in range(N)]
ax_net.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
ax_net.set_ylim(min(ys) - 0.8, max(ys) + 0.6)

# Edges — glow effect
for (u, v) in EDGES:
    xu, yu = POS[u]; xv, yv = POS[v]
    ax_net.plot([xu, xv], [yu, yv], '-', color='#2a4a6a', lw=5.0, alpha=0.35, zorder=1)
    ax_net.plot([xu, xv], [yu, yv], '-', color='#6699bb', lw=1.8, alpha=0.85, zorder=2)

# Node sizes
node_sz = [900 if ROLES[n] == 'Hub'
           else 580 if 'Sub' in ROLES[n]
           else 420 if 'Non-Hub' in ROLES[n]
           else 300 for n in range(N)]
node_x = [POS[n][0] for n in range(N)]
node_y = [POS[n][1] for n in range(N)]
node_c = [ROLE_COLORS[ROLES[n]] for n in range(N)]

# Glow halo
ax_net.scatter(node_x, node_y, s=[s * 2.5 for s in node_sz],
               c=node_c, alpha=0.15, edgecolors='none', zorder=3)
# Main nodes
ax_net.scatter(node_x, node_y, s=node_sz, c=node_c,
               edgecolors='white', linewidths=1.8, zorder=4)

# Label offsets per node to avoid overlap
LABEL_OFFSET = {
    0: (0, -0.52),
    1: (-0.55, 0),  2: (-0.55, 0),
    3: (0, 0.45),   4: (0.55, 0.1), 5: (0.55, 0),
    6: (-0.55, 0),  7: (-0.55, 0),
    8: (-0.55, 0),  9: (0, 0.45),
    10: (0.55, 0),  11: (0.55, 0),
}

for n in range(N):
    x, y = POS[n]
    # Node number inside circle
    ax_net.text(x, y, str(n), ha='center', va='center',
                color='white', fontsize=10, fontweight='bold', zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])
    # L₀ label offset from node
    ox, oy = LABEL_OFFSET[n]
    ax_net.text(x + ox, y + oy, f'L₀={L0[n]:.1f}',
                ha='center', va='center',
                color='#aabbcc', fontsize=8.5, zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])

# Legend — bottom centre of network panel
legend_elements = [
    Patch(facecolor=ROLE_COLORS['Hub'],       edgecolor='white', label='Hub  (node 0)'),
    Patch(facecolor=ROLE_COLORS['Provincial Hub A'],  edgecolor='white', label='Provincial Hub  (1–2)'),
    Patch(facecolor=ROLE_COLORS['Non-Hub Connector'], edgecolor='white', label='Non-Hub Connector  (3–5)'),
    Patch(facecolor=ROLE_COLORS['Leaf'],              edgecolor='white', label='Leaf  (6–11)'),
]
ax_net.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.04),
              ncol=4, facecolor='#0f0f1a', edgecolor='#334455',
              labelcolor='white', fontsize=9.5, framealpha=0.95,
              borderpad=0.7, handlelength=1.2, columnspacing=1.0)

out = os.path.join(FIGS_DIR, 'ml_00_matrix_and_network.png')
plt.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
