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
# ---
# # Section 3 — Motter–Lai Cascade Failure Animation
#
# **Library used:** [NDLib](https://github.com/GiulioRossetti/ndlib) — Network Diffusion Library
# by Rossetti et al. (2018). NDLib provides a `ThresholdModel` that maps directly onto
# Motter-Lai: each node fails when its cumulative load from neighbours exceeds its capacity.
#
# We build a small 12-node network, run the NDLib threshold cascade, then animate it
# with matplotlib showing each round of failure in sequence.
#
# **Why no dedicated Motter-Lai pip package?**
# The original Motter & Lai (2002) model is implemented directly on NetworkX betweenness
# centrality — every published implementation is a ~50-line custom function. NDLib's
# `ThresholdModel` is the closest published library equivalent, modelling node failure
# when load from failed neighbours exceeds a threshold (capacity).

# %%
import os, sys
os.environ['VTK_DEFAULT_RENDER_WINDOW_TYPE'] = 'OSMesa'

# Allow running as a script directly (python src/example_motter_lai/visualization.py)
# as well as imported as a package (from example_motter_lai import visualization)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import numpy as np
import networkx as nx
from IPython.display import Image, display

from example_motter_lai.model import (
    EDGES, ROLES, POS, N, ROLE_COLORS, ALPHA,
    build_network, compute_loads_capacities,
    run_ndlib_cascade, build_frame_data,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

np.random.seed(42)

# %% [markdown]
# ## 3A — Build the Network & Run NDLib Threshold Cascade

# %%
G = build_network()
L0, C = compute_loads_capacities(G)

print('Initial betweenness loads (L0):')
for n in range(N):
    print(f'  Node {n:2d} ({ROLES[n]:<12}) L0={L0[n]:.2f}  C={C[n]:.2f}')

rounds = run_ndlib_cascade(G, C)
frame_data, HOLD, TRANS = build_frame_data(rounds)

FPS = 8
TOTAL_FRAMES = len(frame_data)
print(f'\nNDLib cascade: {len(rounds)} distinct rounds')
for ri, r in enumerate(rounds):
    failed = [n for n, s in r['status'].items() if s == 1]
    print(f'  Round {ri}: failed nodes = {failed}  ({len(failed)}/{N})')
print(f'Total animation frames: {TOTAL_FRAMES}')

# %% [markdown]
# ## 3B — Animate the Cascade

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                          gridspec_kw={'width_ratios': [1.6, 1]})
fig.patch.set_facecolor('#0a0a12')

ax_net, ax_info = axes
for ax in axes:
    ax.set_facecolor('#0a0a12')
ax_net.set_aspect('equal')
ax_net.axis('off')

fig.suptitle(
    'Motter–Lai Cascade Failure  |  NDLib ThresholdModel  |  α = 0.3  |  Attack: Hub (Node 0)',
    color='white', fontsize=12, fontweight='bold', y=0.98)

# ── Draw static edges ─────────────────────────────────────────────────────────
edge_lines = {}
for (u, v) in EDGES:
    xu, yu = POS[u]; xv, yv = POS[v]
    line, = ax_net.plot([xu, xv], [yu, yv], '-', color='#445566',
                         lw=1.8, alpha=0.7, zorder=1)
    edge_lines[(u, v)] = line
    edge_lines[(v, u)] = line

# ── Draw nodes ────────────────────────────────────────────────────────────────
node_x = np.array([POS[n][0] for n in range(N)])
node_y = np.array([POS[n][1] for n in range(N)])
node_sz = np.array([400 if ROLES[n] == 'Hub'
                    else 280 if 'Sub' in ROLES[n]
                    else 220 if 'Non-Hub' in ROLES[n]
                    else 160 for n in range(N)])

scat = ax_net.scatter(node_x, node_y, s=node_sz,
                       c=[ROLE_COLORS[ROLES[n]] for n in range(N)],
                       edgecolors='#cccccc', linewidths=1.0, zorder=4)

for n in range(N):
    x, y = POS[n]
    ax_net.text(x, y - 0.38, f'{ROLES[n]}\nL={L0[n]:.1f}',
                ha='center', va='top', color='#cccccc', fontsize=7, zorder=6,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

load_arts = []
for n in range(N):
    x, y = POS[n]
    t = ax_net.text(x, y, f'{L0[n]/C[n]:.2f}' if C[n] > 1e-9 else '0',
                    ha='center', va='center', color='white',
                    fontsize=8, fontweight='bold', zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
    load_arts.append(t)

# ── Right panel: round tracker + G bar ────────────────────────────────────────
ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
ax_info.axis('off')
ax_info.text(0.5, 0.97, 'CASCADE MONITOR', ha='center', va='top',
             color='#7ec8f0', fontsize=13, fontweight='bold')

N_ROUNDS = len(rounds)
round_boxes, round_texts = [], []
for ri in range(N_ROUNDS):
    y = 0.88 - ri * 0.12
    bg = FancyBboxPatch((0.05, y - 0.045), 0.90, 0.09,
                          boxstyle='round,pad=0.01',
                          facecolor='#1a1a2a', edgecolor='#334455', lw=1.0)
    ax_info.add_patch(bg)
    round_boxes.append(bg)
    txt = ax_info.text(0.50, y, f'Round {ri}: —', ha='center', va='center',
                        color='#666677', fontsize=9, fontweight='bold')
    round_texts.append(txt)

ax_info.text(0.05, 0.20, 'Network Survival  G = N\'/N', color='#aaaaaa', fontsize=9)
g_bg = FancyBboxPatch((0.05, 0.12), 0.90, 0.055,
                        boxstyle='round,pad=0.005', facecolor='#1a1a2a', edgecolor='#334455', lw=1.0)
ax_info.add_patch(g_bg)
g_fill = FancyBboxPatch((0.05, 0.12), 0.90, 0.055,
                          boxstyle='round,pad=0.005', facecolor='#27ae60', edgecolor='none')
ax_info.add_patch(g_fill)
g_val_text = ax_info.text(0.50, 0.148, 'G = 1.000  (12/12)', ha='center', va='center',
                            color='white', fontsize=9, fontweight='bold')

status_text = ax_info.text(0.50, 0.04, 'Healthy Network', ha='center', va='center',
                             color='white', fontsize=10, fontweight='bold',
                             bbox=dict(facecolor='#1a1a2a', edgecolor='#446688', pad=5))


def update(fi):
    ri, sub, is_trans = frame_data[fi]
    curr_status = rounds[ri]['status']
    next_status = rounds[ri + 1]['status'] if ri < len(rounds) - 1 else curr_status

    failed_now = set(n for n, s in curr_status.items() if s == 1)
    new_failing = set(n for n, s in next_status.items() if s == 1) - failed_now if is_trans else set()

    colors, sizes, ec = [], [], []
    for n in range(N):
        if n in failed_now:
            colors.append('#1a1a2a'); sizes.append(node_sz[n] * 0.3); ec.append('#333344')
        elif n in new_failing:
            pulse = 0.5 + 0.5 * np.sin(sub * 2.0)
            colors.append((1.0, 0.2 * pulse, 0.1 * pulse, 1.0))
            sizes.append(node_sz[n] * (1.4 + 0.3 * pulse)); ec.append('#ffff00')
        else:
            colors.append(ROLE_COLORS[ROLES[n]]); sizes.append(node_sz[n]); ec.append('#cccccc')
    scat.set_facecolor(colors); scat.set_sizes(sizes); scat.set_edgecolors(ec)

    active = [n for n in range(N) if n not in failed_now]
    if len(active) >= 2:
        sub_g = G.subgraph(active).copy()
        load_map = nx.betweenness_centrality(sub_g, normalized=False)
    else:
        load_map = {}
    for n, art in enumerate(load_arts):
        if n in failed_now:
            art.set_text('✕'); art.set_color('#ff3333')
        else:
            frac = load_map.get(n, 0.0) / C[n] if C[n] > 1e-9 else 0.0
            art.set_text(f'{frac:.2f}')
            art.set_color('#ff4444' if frac > 1.0 else 'white')

    for (u, v), line in edge_lines.items():
        if u < v:
            if u in failed_now or v in failed_now:
                line.set_alpha(0.06); line.set_color('#1a1a2a')
            elif u in new_failing or v in new_failing:
                line.set_alpha(0.9); line.set_color('#ff6600')
            else:
                line.set_alpha(0.55); line.set_color('#445566')

    for ri2, (box, txt) in enumerate(zip(round_boxes, round_texts)):
        if ri2 < ri:
            box.set_edgecolor('#e74c3c'); box.set_facecolor('#1a0505')
            f_cnt = sum(1 for s in rounds[ri2]['status'].values() if s == 1)
            txt.set_text(f'Round {ri2}: {f_cnt} failed'); txt.set_color('#e74c3c')
        elif ri2 == ri:
            box.set_edgecolor('#ffdd00'); box.set_facecolor('#1a1a05')
            txt.set_text(f'▶ Round {ri2}: {len(failed_now)} failed ({N - len(failed_now)} alive)')
            txt.set_color('#ffdd00')
        else:
            box.set_edgecolor('#334455'); box.set_facecolor('#1a1a2a')
            txt.set_text(f'Round {ri2}: pending…'); txt.set_color('#445566')

    G_val = (N - len(failed_now)) / N
    g_fill.set_width(0.90 * G_val)
    g_fill.set_facecolor('#27ae60' if G_val > 0.7 else '#f39c12' if G_val > 0.4 else '#e74c3c')
    g_val_text.set_text(f'G = {G_val:.3f}  ({N - len(failed_now)}/{N} alive)')

    if len(failed_now) == 0:
        stxt, scol = 'Healthy Network', '#27ae60'
    elif ri == 0:
        stxt, scol = '⚡ Hub Attacked! (Node 0)', '#ffdd00'
    elif is_trans:
        stxt, scol = f'⚠ {len(new_failing)} node(s) overloaded — cascading', '#ff6600'
    else:
        mode = 'CATASTROPHIC' if G_val < 0.2 else 'CASCADING' if G_val < 0.6 else 'CONTAINED'
        stxt, scol = f'Round {ri} stable — {mode}', '#e74c3c' if G_val < 0.4 else '#f39c12'
    status_text.set_text(stxt); status_text.set_color(scol)

    return [scat] + load_arts + list(edge_lines.values()) + \
           round_boxes + round_texts + [g_fill, g_val_text, status_text]


anim_ml = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                                    interval=1000 // FPS, blit=True)
OUT = os.path.join(FIGS_DIR, 'example_motter_lai.gif')
anim_ml.save(OUT, writer=animation.PillowWriter(fps=FPS), dpi=130)
plt.close(fig)
print(f'Saved: {OUT}')

# %%
display(Image(OUT))
