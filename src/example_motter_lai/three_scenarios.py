"""Generate three scenario figures: network legend + three cascade outcomes."""
import os, sys
os.environ['VTK_DEFAULT_RENDER_WINDOW_TYPE'] = 'OSMesa'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import networkx as nx

from example_motter_lai.model import EDGES, ROLES, POS, N, ROLE_COLORS

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

np.random.seed(42)

BG = '#0a0a12'

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Static network + role definitions
# ─────────────────────────────────────────────────────────────────────────────
def make_legend_figure():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                              gridspec_kw={'width_ratios': [1.4, 1]})
    fig.patch.set_facecolor(BG)
    ax_net, ax_def = axes
    for ax in axes:
        ax.set_facecolor(BG)
    ax_net.set_aspect('equal')
    ax_net.axis('off')

    fig.suptitle('Motter–Lai Network — Role Definitions', color='white',
                 fontsize=14, fontweight='bold', y=0.98)

    # Draw edges
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(EDGES)
    for (u, v) in EDGES:
        xu, yu = POS[u]; xv, yv = POS[v]
        ax_net.plot([xu, xv], [yu, yv], '-', color='#445566', lw=2.0, alpha=0.7, zorder=1)

    # Draw nodes
    node_sz = [500 if ROLES[n] == 'Hub'
               else 320 if 'Sub' in ROLES[n]
               else 250 if 'Non-Hub' in ROLES[n]
               else 180 for n in range(N)]
    colors = [ROLE_COLORS[ROLES[n]] for n in range(N)]
    node_x = [POS[n][0] for n in range(N)]
    node_y = [POS[n][1] for n in range(N)]
    ax_net.scatter(node_x, node_y, s=node_sz, c=colors,
                   edgecolors='white', linewidths=1.2, zorder=4)

    # Node labels
    for n in range(N):
        x, y = POS[n]
        ax_net.text(x, y, str(n), ha='center', va='center',
                    color='white', fontsize=9, fontweight='bold', zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        ax_net.text(x, y - 0.42, ROLES[n], ha='center', va='top',
                    color='#cccccc', fontsize=7, zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    # ── Right panel: definitions ───────────────────────────────────────────────
    ax_def.set_xlim(0, 1); ax_def.set_ylim(0, 1)
    ax_def.axis('off')

    ax_def.text(0.5, 0.97, 'NODE ROLE DEFINITIONS', ha='center', va='top',
                color='#7ec8f0', fontsize=13, fontweight='bold')

    definitions = [
        ('Hub', '#ff6600', 'Node 0',
         'Central relay — highest betweenness\n'
         'centrality. Nearly ALL shortest paths\n'
         'cross here. Removing it forces all\n'
         'traffic to reroute through sub-hubs.'),
        ('Provincial Hub', '#f39c12', 'Nodes 1–2',
         'Secondary relays — moderate betweenness.\n'
         'Carry regional traffic between the hub\n'
         'and their leaf clusters. Fail after hub\n'
         'removal if overloaded.'),
        ('Non-Hub Connector\n(mod-high PC)', '#27ae60', 'Nodes 3–5',
         'Cross-region connectors — low-moderate\n'
         'betweenness. Link the hub to far leaf\n'
         'groups. Vulnerable if multiple paths\n'
         'suddenly reroute through them.'),
        ('Leaf\n(low PC)', '#3498db', 'Nodes 6–11',
         'Peripheral endpoints — lowest betweenness.\n'
         'Traffic mostly passes around them.\n'
         'Rarely overloaded even after hub\n'
         'removal; last to fail.'),
    ]

    y = 0.87
    for role, color, nodes, desc in definitions:
        # Colored box header
        bg = FancyBboxPatch((0.03, y - 0.01), 0.94, 0.115,
                             boxstyle='round,pad=0.01',
                             facecolor='#111122', edgecolor=color, lw=1.5)
        ax_def.add_patch(bg)
        ax_def.scatter([0.10], [y + 0.075], s=160, c=[color],
                       edgecolors='white', linewidths=0.8, zorder=5)
        ax_def.text(0.18, y + 0.075, f'{role}  ({nodes})',
                    va='center', color=color, fontsize=10, fontweight='bold')
        ax_def.text(0.06, y + 0.008, desc, va='top', color='#aaaaaa', fontsize=8,
                    linespacing=1.4)
        y -= 0.225

    # Betweenness note
    ax_def.text(0.5, 0.05,
                'Load = betweenness centrality\n'
                'Capacity C = (1 + α) × initial load    (α = 0.3)',
                ha='center', va='center', color='#667788', fontsize=9,
                style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIGS_DIR, 'ml_01_network_legend.png')
    plt.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f'Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Cascade engine — custom betweenness-based Motter-Lai (not NDLib)
# so we can control alpha and which node is attacked
# ─────────────────────────────────────────────────────────────────────────────
def run_motter_lai(G, alpha, attack_node):
    """Pure Motter-Lai: betweenness load, capacity = (1+alpha)*L0."""
    L0 = nx.betweenness_centrality(G, normalized=False)
    C = {n: (1 + alpha) * L0[n] for n in G.nodes()}

    failed = {attack_node}
    rounds = [set()]          # round 0: just attack
    rounds[0].add(attack_node)

    cumulative_failed = set(failed)
    for _ in range(30):
        active = [n for n in G.nodes() if n not in cumulative_failed]
        if len(active) < 2:
            break
        sub = G.subgraph(active).copy()
        load = nx.betweenness_centrality(sub, normalized=False)
        new_fail = {n for n in active if load.get(n, 0) > C[n]}
        if not new_fail:
            break
        rounds.append(new_fail)
        cumulative_failed |= new_fail

    return L0, C, rounds


def make_scenario_gif(label, alpha, attack_node, tag):
    """Build a scenario GIF with the given parameters."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(EDGES)

    L0, C, rounds = run_motter_lai(G, alpha, attack_node)

    # Build cumulative failed-set per round
    cumulative = []
    cum = set()
    for r in rounds:
        cum = cum | r
        cumulative.append(frozenset(cum))

    total_failed = len(cumulative[-1])
    G_val_final = (N - total_failed) / N
    if G_val_final > 0.75:
        outcome_label = 'CONTAINED'
        outcome_color = '#27ae60'
    elif G_val_final > 0.50:
        outcome_label = 'FAILING'
        outcome_color = '#f39c12'
    else:
        outcome_label = 'CATASTROPHIC'
        outcome_color = '#e74c3c'

    # ── Frames: HOLD hold-frames per round, TRANS transition frames ────────────
    # fps=2, so HOLD=14 → 7 seconds per round; TRANS=10 → 5 seconds transition
    HOLD, TRANS = 14, 10
    frame_data = []
    for ri in range(len(rounds)):
        for f in range(HOLD):
            frame_data.append((ri, f, False))
        if ri < len(rounds) - 1:
            for f in range(TRANS):
                frame_data.append((ri, f, True))

    node_sz_base = np.array([500 if ROLES[n] == 'Hub'
                              else 320 if 'Sub' in ROLES[n]
                              else 250 if 'Non-Hub' in ROLES[n]
                              else 180 for n in range(N)])

    fig, (ax_net, ax_info) = plt.subplots(1, 2, figsize=(14, 7),
                                           gridspec_kw={'width_ratios': [1.5, 1]})
    fig.patch.set_facecolor(BG)
    for ax in (ax_net, ax_info):
        ax.set_facecolor(BG)
    ax_net.set_aspect('equal')
    ax_net.axis('off')

    title = (f'Motter–Lai Cascade  |  α={alpha}  |  Attack: {ROLES[attack_node]} '
             f'(Node {attack_node})  →  {outcome_label}')
    fig.suptitle(title, color=outcome_color, fontsize=11, fontweight='bold', y=0.98)

    # Static edges
    edge_lines = {}
    for (u, v) in EDGES:
        xu, yu = POS[u]; xv, yv = POS[v]
        line, = ax_net.plot([xu, xv], [yu, yv], '-', color='#445566',
                             lw=1.8, alpha=0.7, zorder=1)
        edge_lines[(u, v)] = line
        edge_lines[(v, u)] = line

    node_x = np.array([POS[n][0] for n in range(N)])
    node_y = np.array([POS[n][1] for n in range(N)])
    scat = ax_net.scatter(node_x, node_y, s=node_sz_base,
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

    # Right panel
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.axis('off')
    ax_info.text(0.5, 0.97, 'CASCADE MONITOR', ha='center', va='top',
                 color='#7ec8f0', fontsize=13, fontweight='bold')

    N_ROUNDS = len(rounds)
    round_boxes, round_texts = [], []
    for ri in range(N_ROUNDS):
        y_pos = 0.88 - ri * min(0.12, 0.75 / N_ROUNDS)
        bg = FancyBboxPatch((0.05, y_pos - 0.045), 0.90, 0.085,
                             boxstyle='round,pad=0.01',
                             facecolor='#1a1a2a', edgecolor='#334455', lw=1.0)
        ax_info.add_patch(bg)
        round_boxes.append(bg)
        txt = ax_info.text(0.50, y_pos, f'Round {ri}: —', ha='center', va='center',
                            color='#666677', fontsize=9, fontweight='bold')
        round_texts.append(txt)

    ax_info.text(0.05, 0.20, "Network Survival  G = N'/N",
                 color='#aaaaaa', fontsize=9)
    g_bg = FancyBboxPatch((0.05, 0.12), 0.90, 0.055,
                            boxstyle='round,pad=0.005',
                            facecolor='#1a1a2a', edgecolor='#334455', lw=1.0)
    ax_info.add_patch(g_bg)
    g_fill = FancyBboxPatch((0.05, 0.12), 0.90, 0.055,
                              boxstyle='round,pad=0.005',
                              facecolor='#27ae60', edgecolor='none')
    ax_info.add_patch(g_fill)
    g_val_text = ax_info.text(0.50, 0.148, f'G = 1.000  ({N}/{N} alive)',
                               ha='center', va='center', color='white',
                               fontsize=9, fontweight='bold')

    status_text = ax_info.text(0.50, 0.04, 'Healthy Network', ha='center', va='center',
                                color='white', fontsize=10, fontweight='bold',
                                bbox=dict(facecolor='#1a1a2a', edgecolor='#446688', pad=5))

    def update(fi):
        ri, sub, is_trans = frame_data[fi]
        failed_now = cumulative[ri]
        next_failed = cumulative[ri + 1] if ri < len(rounds) - 1 else failed_now
        new_failing = (next_failed - failed_now) if is_trans else set()

        colors, sizes, ec = [], [], []
        for n in range(N):
            if n in failed_now:
                colors.append('#1a1a2a'); sizes.append(node_sz_base[n] * 0.3); ec.append('#333344')
            elif n in new_failing:
                pulse = 0.5 + 0.5 * np.sin(sub * 2.0)
                colors.append((1.0, 0.2 * pulse, 0.1 * pulse, 1.0))
                sizes.append(node_sz_base[n] * (1.4 + 0.3 * pulse)); ec.append('#ffff00')
            else:
                colors.append(ROLE_COLORS[ROLES[n]]); sizes.append(node_sz_base[n]); ec.append('#cccccc')
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
            failed_in_round = len(cumulative[ri2])
            if ri2 < ri:
                box.set_edgecolor('#e74c3c'); box.set_facecolor('#1a0505')
                txt.set_text(f'Round {ri2}: {failed_in_round} failed total')
                txt.set_color('#e74c3c')
            elif ri2 == ri:
                box.set_edgecolor('#ffdd00'); box.set_facecolor('#1a1a05')
                alive = N - len(failed_now)
                txt.set_text(f'▶ Round {ri2}: {len(failed_now)} failed ({alive} alive)')
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
            stxt, scol = f'⚡ {ROLES[attack_node]} Attacked! (Node {attack_node})', '#ffdd00'
        elif is_trans:
            stxt, scol = f'⚠ {len(new_failing)} node(s) overloaded — cascading', '#ff6600'
        else:
            mode = 'CATASTROPHIC' if G_val < 0.50 else 'FAILING' if G_val < 0.75 else 'CONTAINED'
            col = '#e74c3c' if G_val < 0.50 else '#f39c12' if G_val < 0.75 else '#27ae60'
            stxt, scol = f'Round {ri} stable — {mode}', col
        status_text.set_text(stxt); status_text.set_color(scol)

        return ([scat] + load_arts + list(edge_lines.values()) +
                round_boxes + round_texts + [g_fill, g_val_text, status_text])

    FPS = 2
    anim = animation.FuncAnimation(fig, update, frames=len(frame_data),
                                    interval=1000 // FPS, blit=True)
    out = os.path.join(FIGS_DIR, f'ml_{tag}.gif')
    anim.save(out, writer=animation.PillowWriter(fps=FPS), dpi=120)
    plt.close(fig)
    print(f'Saved: {out}  (rounds={N_ROUNDS}, final G={G_val_final:.2f}, outcome={outcome_label})')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Figure 1: Network Legend ===')
    make_legend_figure()

    print('\n=== Scenario A: CONTAINED — attack a Leaf (Node 6), α=0.3 ===')
    make_scenario_gif('Contained leaf attack α=0.3', alpha=0.3, attack_node=6, tag='02_contained')

    print('\n=== Scenario B: FAILING — attack Provincial Hub (Node 1), α=0.3 ===')
    make_scenario_gif('Failing provincial hub attack α=0.3', alpha=0.3, attack_node=1, tag='03_failing')

    print('\n=== Scenario C: CATASTROPHIC — attack Hub (Node 0), α=0.05 ===')
    make_scenario_gif('Catastrophic hub attack α=0.05', alpha=0.05, attack_node=0, tag='04_catastrophic')

    print('\nDone.')
