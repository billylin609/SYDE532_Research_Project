"""Static final-state images for the three cascade scenarios."""
import os, sys
os.environ['VTK_DEFAULT_RENDER_WINDOW_TYPE'] = 'OSMesa'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Patch, FancyArrowPatch
import numpy as np
import networkx as nx

from example_motter_lai.model import EDGES, ROLES, POS, N, ROLE_COLORS, ALPHA

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

BG    = '#0a0a12'
PANEL = '#0f0f1a'

SCENARIOS = [
    dict(label='CONTAINED',     alpha=0.30, attack=6,  tag='05_result_contained',     color='#27ae60'),
    dict(label='FAILING',       alpha=0.30, attack=1,  tag='06_result_failing',        color='#f39c12'),
    dict(label='CATASTROPHIC',  alpha=0.05, attack=0,  tag='07_result_catastrophic',   color='#e74c3c'),
]


def run_cascade(alpha, attack_node):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(EDGES)
    L0 = nx.betweenness_centrality(G, normalized=False)
    C  = {n: (1 + alpha) * L0[n] for n in G.nodes()}

    cumulative = {attack_node}
    rounds = [frozenset({attack_node})]
    for _ in range(30):
        active = [n for n in G.nodes() if n not in cumulative]
        if len(active) < 2:
            break
        sub  = G.subgraph(active).copy()
        load = nx.betweenness_centrality(sub, normalized=False)
        new  = {n for n in active if load.get(n, 0) > C[n]}
        if not new:
            break
        cumulative |= new
        rounds.append(frozenset(new))
    return G, L0, C, rounds, frozenset(cumulative)


def draw_scenario(sc):
    alpha, attack, label, color, tag = (
        sc['alpha'], sc['attack'], sc['label'], sc['color'], sc['tag'])

    G, L0, C, rounds, failed = run_cascade(alpha, attack)
    survived = [n for n in range(N) if n not in failed]
    G_val = len(survived) / N

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.axis('off')

    # ── Title block ──────────────────────────────────────────────────────────
    fig.suptitle(f'Cascade Result  —  {label}', color=color,
                 fontsize=17, fontweight='bold', y=0.97)
    ax.set_title(
        f'Attack: {ROLES[attack]} (Node {attack})   |   α = {alpha}   |   '
        f'G = {G_val:.2f}  ({len(survived)}/{N} survived)',
        color='#aabbcc', fontsize=10, pad=10)

    # ── Set axis limits with padding ─────────────────────────────────────────
    xs = [POS[n][0] for n in range(N)]
    ys = [POS[n][1] for n in range(N)]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 0.8)

    # ── Edges ────────────────────────────────────────────────────────────────
    for (u, v) in EDGES:
        xu, yu = POS[u]; xv, yv = POS[v]
        both_alive = u not in failed and v not in failed
        ec = '#5588aa' if both_alive else '#1e2530'
        ew = 2.0      if both_alive else 1.0
        ea = 0.85     if both_alive else 0.25
        ax.plot([xu, xv], [yu, yv], '-', color=ec, lw=ew, alpha=ea, zorder=1)
        if both_alive:
            ax.plot([xu, xv], [yu, yv], '-', color='#2a4a6a',
                    lw=5.0, alpha=0.3, zorder=0)

    # ── Nodes ────────────────────────────────────────────────────────────────
    node_sz_base = {n: (900 if ROLES[n] == 'Hub'
                        else 580 if 'Sub' in ROLES[n]
                        else 420 if 'Non-Hub' in ROLES[n]
                        else 300) for n in range(N)}

    for n in range(N):
        x, y = POS[n]
        sz   = node_sz_base[n]
        if n in failed:
            # Faded ghost node
            ax.scatter(x, y, s=sz * 0.55, c='#1a1a2a',
                       edgecolors='#333344', linewidths=1.2, zorder=3)
            ax.text(x, y, '✕', ha='center', va='center',
                    color='#cc2222', fontsize=13, fontweight='bold', zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        else:
            rc = ROLE_COLORS[ROLES[n]]
            # Glow halo
            ax.scatter(x, y, s=sz * 2.2, c=[rc], alpha=0.15,
                       edgecolors='none', zorder=2)
            ax.scatter(x, y, s=sz, c=[rc],
                       edgecolors='white', linewidths=1.8, zorder=4)
            ax.text(x, y, str(n), ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold', zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

    # ── L₀ / status labels ───────────────────────────────────────────────────
    OFFSETS = {
        0: (0, -0.52),
        1: (-0.60, 0.1), 2: (-0.60, -0.1),
        3: (0.15, 0.48), 4: (0.60, 0.1),  5: (0.60, 0),
        6: (-0.60, 0),   7: (-0.60, 0),
        8: (-0.60, 0),   9: (0.15, 0.48),
        10: (0.60, 0),   11: (0.60, 0),
    }
    for n in range(N):
        x, y = POS[n]
        ox, oy = OFFSETS[n]
        if n in failed:
            txt = 'FAILED'
            tc  = '#cc3333'
        else:
            txt = f'L₀={L0[n]:.1f}'
            tc  = '#99aabb'
        ax.text(x + ox, y + oy, txt, ha='center', va='center',
                color=tc, fontsize=8, zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])


    # ── G-bar at bottom ───────────────────────────────────────────────────────
    bar_y  = min(ys) - 0.68
    bar_x0 = min(xs)
    bar_w  = max(xs) - min(xs)
    ax.add_patch(FancyBboxPatch((bar_x0, bar_y), bar_w, 0.22,
                                 boxstyle='round,pad=0.04',
                                 facecolor='#1a1a2a', edgecolor='#334455', lw=1,
                                 zorder=8))
    ax.add_patch(FancyBboxPatch((bar_x0, bar_y), bar_w * G_val, 0.22,
                                 boxstyle='round,pad=0.04',
                                 facecolor=color, edgecolor='none',
                                 zorder=9))
    ax.text(bar_x0 + bar_w / 2, bar_y + 0.11,
            f"Network Survival  G = {G_val:.2f}  ({len(survived)}/{N} nodes alive)",
            ha='center', va='center', color='white',
            fontsize=9.5, fontweight='bold', zorder=10,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')])


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIGS_DIR, f'{tag}.png')
    plt.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    for sc in SCENARIOS:
        draw_scenario(sc)
    print('Done.')
