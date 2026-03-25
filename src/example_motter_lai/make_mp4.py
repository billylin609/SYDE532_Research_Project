"""Generate three cascade MP4s — animates cascade then freezes on result-style final frame."""
import os, sys
os.environ['VTK_DEFAULT_RENDER_WINDOW_TYPE'] = 'OSMesa'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import networkx as nx

from example_motter_lai.model import EDGES, ROLES, POS, N, ROLE_COLORS

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

BG    = '#0a0a12'
PANEL = '#0f0f1a'

FPS        = 24       # smooth MP4
# Max 10 sec total: worst case = 3 rounds × HOLD + 2 transitions × TRANS + FREEZE
# 3×2 + 2×1 + 2 = 10 sec
HOLD_SEC   = 2        # seconds to hold each round state
TRANS_SEC  = 1        # seconds for transition pulse
FREEZE_SEC = 2        # seconds to hold final result frame
HOLD  = FPS * HOLD_SEC
TRANS = FPS * TRANS_SEC
FREEZE= FPS * FREEZE_SEC

SCENARIOS = [
    dict(label='CONTAINED',    alpha=0.30, attack=6, color='#27ae60',
         tag='ml_08_contained'),
    dict(label='FAILING',      alpha=0.30, attack=1, color='#f39c12',
         tag='ml_09_failing'),
    dict(label='CATASTROPHIC', alpha=0.05, attack=0, color='#e74c3c',
         tag='ml_10_catastrophic'),
]

node_sz_base = np.array([900 if ROLES[n] == 'Hub'
                          else 580 if 'Provincial' in ROLES[n]
                          else 420 if 'Non-Hub' in ROLES[n]
                          else 300 for n in range(N)])


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


def make_mp4(sc):
    label, alpha, attack, color, tag = (
        sc['label'], sc['alpha'], sc['attack'], sc['color'], sc['tag'])

    G, L0, C, rounds, final_failed = run_cascade(alpha, attack)
    survived = [n for n in range(N) if n not in final_failed]
    G_val    = len(survived) / N

    # Cumulative failed per round
    cumulative = []
    cum = set()
    for r in rounds:
        cum = cum | r
        cumulative.append(frozenset(cum))

    # ── Frame schedule ────────────────────────────────────────────────────────
    # [cascade frames] + [freeze on result frame]
    frame_data = []
    for ri in range(len(rounds)):
        for f in range(HOLD):
            frame_data.append(('cascade', ri, f, False))
        if ri < len(rounds) - 1:
            for f in range(TRANS):
                frame_data.append(('cascade', ri, f, True))
    for f in range(FREEZE):
        frame_data.append(('result', len(rounds) - 1, f, False))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.axis('off')
    ax.set_aspect('equal')

    xs = [POS[n][0] for n in range(N)]
    ys = [POS[n][1] for n in range(N)]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 0.8)

    title_txt = fig.suptitle(
        f'Cascade Result  —  {label}  |  α={alpha}  |  Attack: {ROLES[attack]} (Node {attack})',
        color=color, fontsize=15, fontweight='bold', y=0.98)

    # Static edges
    edge_lines = {}
    for (u, v) in EDGES:
        xu, yu = POS[u]; xv, yv = POS[v]
        l1, = ax.plot([xu, xv], [yu, yv], '-', color='#2a4a6a', lw=5.0, alpha=0.3, zorder=1)
        l2, = ax.plot([xu, xv], [yu, yv], '-', color='#5588aa', lw=1.5, alpha=0.8, zorder=2)
        edge_lines[(u, v)] = (l1, l2)
        edge_lines[(v, u)] = (l1, l2)

    # Glow halos
    glow_scat = ax.scatter([POS[n][0] for n in range(N)],
                            [POS[n][1] for n in range(N)],
                            s=node_sz_base * 2.2,
                            c=[ROLE_COLORS[ROLES[n]] for n in range(N)],
                            alpha=0.15, edgecolors='none', zorder=3)
    # Main nodes
    scat = ax.scatter([POS[n][0] for n in range(N)],
                       [POS[n][1] for n in range(N)],
                       s=node_sz_base,
                       c=[ROLE_COLORS[ROLES[n]] for n in range(N)],
                       edgecolors='white', linewidths=1.5, zorder=4)

    # Node labels
    node_arts = []
    for n in range(N):
        x, y = POS[n]
        t = ax.text(x, y, str(n), ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold', zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])
        node_arts.append(t)

    # L0 / status labels
    OFFSETS = {
        0:(0,-0.52), 1:(-0.60,0.1), 2:(-0.60,-0.1),
        3:(0.15,0.48), 4:(0.60,0.1), 5:(0.60,0),
        6:(-0.60,0), 7:(-0.60,0), 8:(-0.60,0),
        9:(0.15,0.48), 10:(0.60,0), 11:(0.60,0),
    }
    label_arts = []
    for n in range(N):
        x, y = POS[n]; ox, oy = OFFSETS[n]
        t = ax.text(x+ox, y+oy, f'L₀={L0[n]:.1f}',
                    ha='center', va='center', color='#99aabb',
                    fontsize=8.5, zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        label_arts.append(t)

    # G-bar
    bar_y  = min(ys) - 0.72
    bar_x0 = min(xs)
    bar_w  = max(xs) - min(xs)
    ax.add_patch(FancyBboxPatch((bar_x0, bar_y), bar_w, 0.24,
                                 boxstyle='round,pad=0.04',
                                 facecolor='#1a1a2a', edgecolor='#334455',
                                 lw=1, zorder=8))
    g_fill = FancyBboxPatch((bar_x0, bar_y), bar_w, 0.24,
                              boxstyle='round,pad=0.04',
                              facecolor='#27ae60', edgecolor='none', zorder=9)
    ax.add_patch(g_fill)
    g_txt = ax.text(bar_x0 + bar_w/2, bar_y + 0.12,
                    f'G = 1.000  ({N}/{N} alive)',
                    ha='center', va='center', color='white',
                    fontsize=10, fontweight='bold', zorder=10,
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    def draw_cascade_frame(ri, sub_f, is_trans):
        """Standard cascade animation frame."""
        failed_now  = cumulative[ri]
        next_failed = cumulative[ri+1] if ri < len(rounds)-1 else failed_now
        new_failing = (next_failed - failed_now) if is_trans else set()

        colors, sizes, ec = [], [], []
        for n in range(N):
            if n in failed_now:
                colors.append('#1a1a2a'); sizes.append(node_sz_base[n]*0.3); ec.append('#333344')
            elif n in new_failing:
                pulse = 0.5 + 0.5*np.sin(sub_f / FPS * 2 * np.pi * 2)
                colors.append((1.0, 0.2*pulse, 0.1*pulse, 1.0))
                sizes.append(node_sz_base[n]*(1.4+0.3*pulse)); ec.append('#ffff00')
            else:
                colors.append(ROLE_COLORS[ROLES[n]]); sizes.append(node_sz_base[n]); ec.append('white')
        scat.set_facecolor(colors); scat.set_sizes(sizes); scat.set_edgecolors(ec)
        glow_scat.set_facecolor(colors)

        for n, art in enumerate(node_arts):
            art.set_text('✕' if n in failed_now else str(n))
            art.set_color('#ff3333' if n in failed_now else 'white')

        for n, art in enumerate(label_arts):
            if n in failed_now:
                art.set_text('FAILED'); art.set_color('#cc3333')
            else:
                art.set_text(f'L₀={L0[n]:.1f}'); art.set_color('#99aabb')

        for (u, v), (l1, l2) in edge_lines.items():
            if u < v:
                if u in failed_now or v in failed_now:
                    l1.set_alpha(0.03); l2.set_alpha(0.06); l2.set_color('#1a1a2a')
                elif u in new_failing or v in new_failing:
                    l1.set_alpha(0.5); l2.set_alpha(0.9); l2.set_color('#ff6600')
                else:
                    l1.set_alpha(0.3); l2.set_alpha(0.8); l2.set_color('#5588aa')

        G_val = (N - len(failed_now)) / N
        g_fill.set_width(bar_w * G_val)
        g_fill.set_facecolor('#27ae60' if G_val > 0.75 else '#f39c12' if G_val > 0.5 else '#e74c3c')
        g_txt.set_text(f'G = {G_val:.3f}  ({N-len(failed_now)}/{N} alive)')

    def draw_result_frame():
        """Final freeze frame — result image style."""
        colors, sizes, ec = [], [], []
        for n in range(N):
            if n in final_failed:
                colors.append('#1a1a2a'); sizes.append(node_sz_base[n]*0.55); ec.append('#333344')
            else:
                colors.append(ROLE_COLORS[ROLES[n]]); sizes.append(node_sz_base[n]); ec.append('white')
        scat.set_facecolor(colors); scat.set_sizes(sizes); scat.set_edgecolors(ec)
        glow_scat.set_facecolor(colors)

        for n, art in enumerate(node_arts):
            art.set_text('✕' if n in final_failed else str(n))
            art.set_color('#cc2222' if n in final_failed else 'white')

        for n, art in enumerate(label_arts):
            art.set_text('FAILED' if n in final_failed else f'L₀={L0[n]:.1f}')
            art.set_color('#cc3333' if n in final_failed else '#99aabb')

        for (u, v), (l1, l2) in edge_lines.items():
            if u < v:
                both_alive = u not in final_failed and v not in final_failed
                l1.set_alpha(0.3 if both_alive else 0.02)
                l2.set_alpha(0.85 if both_alive else 0.06)
                l2.set_color('#5588aa' if both_alive else '#1a1a2a')

        G_val = len(survived) / N
        g_fill.set_width(bar_w * G_val)
        g_fill.set_facecolor(color)
        g_txt.set_text(f'Network Survival  G = {G_val:.2f}  ({len(survived)}/{N} nodes alive)')
        title_txt.set_text(
            f'Cascade Result  —  {label}  |  α={alpha}  |  '
            f'Attack: {ROLES[attack]} (Node {attack})  →  G = {G_val:.2f}')

    def update(fi):
        mode, ri, sub_f, is_trans = frame_data[fi]
        if mode == 'cascade':
            draw_cascade_frame(ri, sub_f, is_trans)
        else:
            draw_result_frame()
        return ([scat, glow_scat, g_fill, g_txt, title_txt]
                + node_arts + label_arts
                + [l for pair in edge_lines.values() for l in pair])

    anim = animation.FuncAnimation(fig, update, frames=len(frame_data),
                                    interval=1000//FPS, blit=True)
    out = os.path.join(FIGS_DIR, f'{tag}.mp4')
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2000,
                                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    anim.save(out, writer=writer, dpi=120)
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    for sc in SCENARIOS:
        print(f'\n=== {sc["label"]} ===')
        make_mp4(sc)
    print('\nDone.')
