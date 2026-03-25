"""Brain cascade MP4s — hub attack at bad α (0.10) and good α (0.40).
Nodes coloured by Leiden community. No header, no G bar.
"""
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np
import networkx as nx

from HCP_MotterLai.model import load_sc_and_loads, motter_lai
from HCP_Community.model import detect_leiden, load_full_sc

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)
np.random.seed(42)

# ── MNI centroid coordinates — DK atlas 68 cortical regions ───────────────────
DK_MNI_LEFT = {
    'bankssts':               (-54, -40),
    'caudalanteriorcingulate':( -8,  12),
    'caudalmiddlefrontal':    (-36,  14),
    'cuneus':                 (-12, -82),
    'entorhinal':             (-24,  -8),
    'fusiform':               (-36, -40),
    'inferiorparietal':       (-46, -56),
    'inferiortemporal':       (-52, -28),
    'isthmuscingulate':       ( -8, -38),
    'lateraloccipital':       (-30, -82),
    'lateralorbitofrontal':   (-24,  34),
    'lingual':                (-14, -66),
    'medialorbitofrontal':    ( -8,  50),
    'middletemporal':         (-58, -18),
    'parahippocampal':        (-22, -28),
    'paracentral':            ( -8, -24),
    'parsopercularis':        (-50,  16),
    'parsorbitalis':          (-32,  34),
    'parstriangularis':       (-44,  30),
    'pericalcarine':          (-10, -76),
    'postcentral':            (-40, -32),
    'posteriorcingulate':     ( -8, -46),
    'precentral':             (-40,  -8),
    'precuneus':              ( -8, -64),
    'rostralanteriorcingulate':(-6,  38),
    'rostralmiddlefrontal':   (-28,  42),
    'superiorfrontal':        (-14,  36),
    'superiorparietal':       (-22, -60),
    'superiortemporal':       (-54, -14),
    'supramarginal':          (-56, -42),
    'frontalpole':            (-10,  62),
    'temporalpole':           (-38,  14),
    'transversetemporal':     (-44, -24),
    'insula':                 (-34,   0),
}


def get_xy(sc_ctx_labels):
    """Return (N, 2) MNI axial (x, y) array for 68 cortical labels."""
    xy = []
    for lbl in sc_ctx_labels:
        s = str(lbl).replace('L_', '').replace('R_', '').lower()
        is_right = str(lbl).startswith('R_')
        x, y = DK_MNI_LEFT.get(s, (0, 0))
        xy.append((-x if is_right else x, y))
    return np.array(xy, dtype=float)


# ── Functional colours ─────────────────────────────────────────────────────────
FUNCTIONAL_ATLAS = {
    'Default Mode Network':    ['posteriorcingulate','isthmuscingulate','precuneus',
                                'medialorbitofrontal','frontalpole','entorhinal',
                                'parahippocampal','middletemporal','inferiortemporal','bankssts'],
    'Sensorimotor':            ['precentral','postcentral','paracentral','superiorfrontal',
                                'rostralmiddlefrontal'],
    'Visual':                  ['cuneus','lingual','pericalcarine','lateraloccipital','fusiform'],
    'Frontoparietal':          ['caudalmiddlefrontal','parsopercularis','parsorbitalis',
                                'parstriangularis','lateralorbitofrontal',
                                'superiorparietal','inferiorparietal','supramarginal'],
    'Temporal / Language':     ['superiortemporal','transversetemporal','temporalpole',
                                'caudalanteriorcingulate'],
    'Limbic':                  ['entorhinal','parahippocampal','temporalpole',
                                'caudalanteriorcingulate','rostralanteriorcingulate'],
    'Salience / Cingulo-Op':   ['insula','rostralanteriorcingulate','caudalanteriorcingulate'],
}
NET_COLORS = {
    'Default Mode Network':    '#E05C5C',
    'Sensorimotor':            '#4A90D9',
    'Visual':                  '#8E44AD',
    'Frontoparietal':          '#27AE60',
    'Temporal / Language':     '#F39C12',
    'Limbic':                  '#1ABC9C',
    'Salience / Cingulo-Op':   '#E67E22',
    'Unassigned':              '#BDC3C7',
}


def assign_net(region_short):
    r = region_short.lower()
    for net, kws in FUNCTIONAL_ATLAS.items():
        if any(k in r for k in kws):
            return net
    return 'Unassigned'


# ── Load HCP data ──────────────────────────────────────────────────────────────
print('Loading HCP SC…')
(sc_ctx, sc_ctx_labels, N, A_bin, G_w, G_uw,
 L0_uw_arr, L0_w_arr, degree_arr) = load_sc_and_loads()

xy           = get_xy(sc_ctx_labels)
short_labels = [str(l).replace('L_', '').replace('R_', '').lower() for l in sc_ctx_labels]
node_net     = [assign_net(s) for s in short_labels]

# ── Leiden community colours — same 82-node run as hcp_brain_partition.png ─────
# Run on full 82-node matrix, then slice first 68 (cortical) labels so colours
# are consistent with the brain partition figure (Q=0.2471, 3 communities).
print('Running Leiden on full 82-node matrix (matching brain partition plot)…')
COMM_PALETTE = ['#E05C5C', '#4A90D9', '#27AE60', '#F39C12', '#8E44AD', '#1ABC9C']
A_full, _, n_full, _, _, _, _, _ = load_full_sc()
labels_leiden_82, Q_leiden = detect_leiden(A_full, n_full)
labels_leiden = labels_leiden_82[:N]          # first 68 = cortical nodes
unique_comms  = np.unique(labels_leiden)
comm_color    = {c: COMM_PALETTE[i % len(COMM_PALETTE)] for i, c in enumerate(unique_comms)}
base_colors   = [comm_color[labels_leiden[i]] for i in range(N)]
print(f'  Leiden Q={Q_leiden:.4f}, {len(unique_comms)} communities (82-node run, first 68 used)')

# Top-8% edge mask for drawing
all_edges  = list(G_w.edges(data=True))
edge_wts   = np.array([d['weight'] for _, _, d in all_edges])
edge_thresh = np.percentile(edge_wts, 92)
draw_edges  = [(u, v) for u, v, d in all_edges if d['weight'] >= edge_thresh]

# ── Braak staging — AD-progression node sets ───────────────────────────────────
def nodes_matching(keywords):
    """Return indices of all 68 regions whose short label contains any keyword."""
    return [i for i, s in enumerate(short_labels) if any(k in s for k in keywords)]

BRAAK_STAGES = [
    ('Braak I–II\n(Transentorhinal)',
     nodes_matching(['entorhinal'])),
    ('Braak III–IV\n(Limbic)',
     nodes_matching(['parahippocampal', 'isthmuscingulate', 'posteriorcingulate'])),
    ('Braak V\n(Early Neocortical)',
     nodes_matching(['inferiortemporal', 'middletemporal', 'fusiform',
                     'lingual', 'pericalcarine', 'lateraloccipital'])),
    ('Braak VI\n(Full Neocortical)',
     nodes_matching(['superiorparietal', 'precuneus', 'superiorfrontal',
                     'caudalmiddlefrontal', 'superiortemporal'])),
]


def run_ad_staged(alpha=0.1):
    """
    Staged Motter-Lai: forcibly remove Braak-stage nodes, then cascade.
    Returns list of dicts with keys: stage_label, forced, cascaded, G_val.
    """
    # Compute initial capacity from full network
    L0 = nx.betweenness_centrality(G_uw, normalized=False)
    C = {i: (1 + alpha) * L0[i] for i in range(N)}

    all_removed = set()
    history = []

    for stage_label, forced_nodes in BRAAK_STAGES:
        # Force remove this stage's nodes (skip already removed)
        forced = frozenset(n for n in forced_nodes if n not in all_removed)
        all_removed |= forced

        # Cascade on remaining
        active = [i for i in range(N) if i not in all_removed]
        cascade_rounds = []
        for _ in range(N):
            if len(active) < 2:
                break
            sub = sc_ctx[np.ix_(active, active)]
            G_sub = nx.from_numpy_array((sub > 0).astype(float))
            L_sub = nx.betweenness_centrality(G_sub, normalized=False)
            overloaded = frozenset(active[j] for j, node in enumerate(active)
                                   if L_sub[j] > C[node])
            if not overloaded:
                break
            active = [i for i in active if i not in overloaded]
            all_removed |= overloaded
            cascade_rounds.append(overloaded)

        if len(active) >= 1:
            G_fin = nx.from_numpy_array((sc_ctx[np.ix_(active, active)] > 0).astype(float))
            lcc = max(nx.connected_components(G_fin), key=len) if G_fin.number_of_nodes() > 0 else set()
            G_val = len(lcc) / N
        else:
            G_val = 0.0

        history.append(dict(stage_label=stage_label, forced=forced,
                            cascade_rounds=cascade_rounds,
                            all_removed=frozenset(all_removed), G_val=G_val))
        print(f'  {stage_label.split(chr(10))[0]}: forced={len(forced)}, '
              f'cascade={sum(len(r) for r in cascade_rounds)}, G={G_val:.3f}')

    return history


# ── Animation parameters ───────────────────────────────────────────────────────
FPS        = 24
HOLD_SEC   = 2
TRANS_SEC  = 1
FREEZE_SEC = 2
HOLD  = FPS * HOLD_SEC
TRANS = FPS * TRANS_SEC
FREEZE = FPS * FREEZE_SEC

BG    = '#0a0a12'
PANEL = '#0f1520'


def build_fig_ax():
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlim(-90, 90)
    ax.set_ylim(-98, 95)

    # Brain outline
    brain = Ellipse((0, -10), width=150, height=175,
                    facecolor=PANEL, edgecolor='#334455', linewidth=1.5, zorder=0)
    ax.add_patch(brain)
    ax.axvline(0, color='#334455', lw=0.8, alpha=0.5, zorder=1)
    for txt, x, y in [('L', -55, 80), ('R', 55, 80),
                       ('Anterior', 0, 85), ('Posterior', 0, -92)]:
        ax.text(x, y, txt, color='#445566', fontsize=9 if len(txt) > 1 else 12,
                fontweight='bold', ha='center', va='center')

    # Static edges
    for u, v in draw_edges:
        ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]],
                '-', color='#2a4a6a', lw=0.7, alpha=0.35, zorder=2)

    # Glow
    glow = ax.scatter(xy[:, 0], xy[:, 1], s=200, c=base_colors,
                      alpha=0.12, edgecolors='none', zorder=3)
    # Nodes
    scat = ax.scatter(xy[:, 0], xy[:, 1], s=50, c=base_colors,
                      edgecolors='white', linewidths=0.5, zorder=4)

    # G-bar background
    bx, by, bw, bh = -60, -90, 120, 6
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                 boxstyle='round,pad=0.4',
                                 facecolor='#1a1a2a', edgecolor='#334455',
                                 lw=1, zorder=8))
    g_fill = FancyBboxPatch((bx, by), bw, bh,
                              boxstyle='round,pad=0.4',
                              facecolor='#27ae60', edgecolor='none', zorder=9)
    ax.add_patch(g_fill)
    g_txt = ax.text(0, by + bh / 2,
                    f'G = 1.000  ({N}/{N} alive)',
                    ha='center', va='center', color='white',
                    fontsize=9, fontweight='bold', zorder=10,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    return fig, ax, scat, glow, g_fill, g_txt, (bx, bw)


def update_scatter(scat, glow, failed_now, pulsing=frozenset(), forced=frozenset(), sub_f=0):
    colors, sizes, ec = [], [], []
    for i in range(N):
        if i in failed_now:
            colors.append('#111122'); sizes.append(18); ec.append('#222233')
        elif i in pulsing:
            pulse = 0.5 + 0.5 * np.sin(sub_f / FPS * 2 * np.pi * 2)
            colors.append((1.0, 0.2 * pulse, 0.05 * pulse, 1.0))
            sizes.append(80 * (1.3 + 0.3 * pulse)); ec.append('#ffff00')
        elif i in forced:                      # forced AD removal: amber pulse
            pulse = 0.5 + 0.5 * np.sin(sub_f / FPS * 2 * np.pi * 2)
            colors.append((1.0, 0.55 + 0.1 * pulse, 0.0, 1.0))
            sizes.append(80 * (1.3 + 0.3 * pulse)); ec.append('#ffffff')
        else:
            colors.append(base_colors[i]); sizes.append(50); ec.append('white')
    scat.set_facecolor(colors); scat.set_sizes(sizes); scat.set_edgecolors(ec)
    glow.set_facecolor(colors)


def update_gbar(g_fill, g_txt, failed_set, bx, bw, accent_color=None, label_override=None):
    G_val = (N - len(failed_set)) / N
    g_fill.set_width(bw * G_val)
    if accent_color:
        g_fill.set_facecolor(accent_color)
    else:
        g_fill.set_facecolor('#27ae60' if G_val > 0.75 else '#f39c12' if G_val > 0.5 else '#e74c3c')
    txt = label_override or f'G = {G_val:.3f}  ({N - len(failed_set)}/{N} alive)'
    g_txt.set_text(txt)


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION — Hub attack, configurable alpha
# ══════════════════════════════════════════════════════════════════════════════
def make_hub_mp4(alpha, tag, accent_color, attack=None):
    """Hub attack at given alpha. Nodes coloured by Leiden community.
    attack: node index (default = argmax betweenness = R_superiorparietal)."""
    if attack is None:
        attack = int(np.argmax(L0_uw_arr))
    attack_name = str(sc_ctx_labels[attack])
    attack_L0   = L0_uw_arr[attack]

    print(f'\n=== HUB ATTACK  α={alpha}  ({tag}) ===')
    G_metric, n_failed, history = motter_lai(sc_ctx, attack, alpha, return_history=True)
    print(f'  G={G_metric:.3f}  failed={n_failed}  rounds={len(history)-1}')

    cumulative = []
    cum = set()
    for r in history:
        cum = cum | r
        cumulative.append(frozenset(cum))
    final_failed = cumulative[-1]
    survived     = [i for i in range(N) if i not in final_failed]

    frame_data = []
    for ri in range(len(history)):
        for f in range(HOLD):
            frame_data.append(('cascade', ri, f, False))
        if ri < len(history) - 1:
            for f in range(TRANS):
                frame_data.append(('cascade', ri, f, True))
    for f in range(FREEZE):
        frame_data.append(('result', len(history) - 1, f, False))

    # ── Figure: full-brain, no header, no G bar ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
    fig.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98)
    ax.set_facecolor(BG); ax.axis('off'); ax.set_aspect('equal')
    ax.set_xlim(-90, 90); ax.set_ylim(-95, 95)

    # Title line (static)
    fig.suptitle(
        f'Hub Attack — {attack_name}  |  α = {alpha}  '
        f'({"cascade failure" if G_metric < 0.85 else "contained"})',
        color=accent_color, fontsize=11, fontweight='bold', y=0.97)

    # Brain outline
    brain = Ellipse((0, -10), width=150, height=175,
                    facecolor=PANEL, edgecolor='#334455', linewidth=1.5, zorder=0)
    ax.add_patch(brain)
    ax.axvline(0, color='#334455', lw=0.8, alpha=0.5, zorder=1)
    for txt, x, y in [('L', -60, 78), ('R', 60, 78),
                       ('Anterior', 0, 83), ('Posterior', 0, -90)]:
        ax.text(x, y, txt, color='#556677',
                fontsize=8 if len(txt) > 1 else 11,
                fontweight='bold', ha='center', va='center')

    # Edges
    for u, v in draw_edges:
        ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]],
                '-', color='#2a4a6a', lw=0.7, alpha=0.35, zorder=2)

    # Leiden community legend (bottom-left, inside brain area)
    for ci, c in enumerate(unique_comms):
        members = [i for i in range(N) if labels_leiden[i] == c]
        ax.scatter([], [], c=comm_color[c], s=40, label=f'C{c}  (n={len(members)})',
                   edgecolors='white', linewidths=0.4)
    ax.legend(loc='lower left', fontsize=7, facecolor='#0d0d1a',
              edgecolor='#334455', labelcolor='white', framealpha=0.85,
              markerscale=1.0, handletextpad=0.4, borderpad=0.5)

    # Glow + nodes
    glow = ax.scatter(xy[:, 0], xy[:, 1], s=220, c=base_colors,
                      alpha=0.12, edgecolors='none', zorder=3)
    scat = ax.scatter(xy[:, 0], xy[:, 1], s=55, c=base_colors,
                      edgecolors='white', linewidths=0.5, zorder=4)

    # Persistent star + label on attack node
    ax.scatter([xy[attack, 0]], [xy[attack, 1]], s=340, marker='*',
               c=accent_color, edgecolors='#ffff00', linewidths=1.0, zorder=7)
    ax.annotate(attack_name.replace('R_', 'R '),
                xy=(xy[attack, 0], xy[attack, 1]),
                xytext=(xy[attack, 0] + 12, xy[attack, 1] + 9),
                color=accent_color, fontsize=7, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')],
                arrowprops=dict(arrowstyle='->', color=accent_color, lw=1.0),
                zorder=8)

    # Dynamic round text (above brain ellipse, no overlap)
    rnd_txt = ax.text(0, 88, 'Round 0  —  Seed removed',
                      ha='center', va='center', fontsize=8.5,
                      color='#f39c12', fontweight='bold', zorder=11,
                      path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    # ── update ────────────────────────────────────────────────────────────────
    def update(fi):
        mode, ri, sub_f, is_trans = frame_data[fi]
        failed_now  = cumulative[ri]
        next_failed = cumulative[ri + 1] if ri < len(history) - 1 else failed_now
        pulsing     = (next_failed - failed_now) if is_trans else frozenset()

        if mode == 'cascade':
            update_scatter(scat, glow, failed_now, pulsing=pulsing, sub_f=sub_f)
            if ri == 0:
                rnd_txt.set_text(f'Round 0  —  Seed removed  ({len(failed_now)} failed)')
            else:
                rnd_txt.set_text(
                    f'Round {ri}  —  +{len(history[ri])} overloaded  '
                    f'({len(failed_now)}/{N} failed)')
        else:
            update_scatter(scat, glow, final_failed)
            rnd_txt.set_text(
                f'Cascade complete — G = {G_metric:.2f}  '
                f'({len(survived)}/{N} survived)')
        return [scat, glow, rnd_txt]

    anim = animation.FuncAnimation(fig, update, frames=len(frame_data),
                                    interval=1000 // FPS, blit=True)
    out = os.path.join(FIGS_DIR, f'{tag}.mp4')
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2000,
                                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    anim.save(out, writer=writer, dpi=130)
    plt.close(fig)
    print(f'Saved: {out}')

    # Also save final result frame as PNG for LaTeX
    _save_result_png(attack, final_failed, survived, G_metric, alpha, accent_color, tag)


def _save_result_png(attack, final_failed, survived, G_metric, alpha, accent_color, tag):
    """Static PNG of the final cascade state."""
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
    fig.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98)
    ax.set_facecolor(BG); ax.axis('off'); ax.set_aspect('equal')
    ax.set_xlim(-90, 90); ax.set_ylim(-95, 95)

    fig.suptitle(
        f'Hub Attack — {sc_ctx_labels[attack]}  |  α = {alpha}  '
        f'({"cascade failure" if G_metric < 0.85 else "contained"})  →  G = {G_metric:.2f}',
        color=accent_color, fontsize=11, fontweight='bold', y=0.97)

    brain = Ellipse((0, -10), width=150, height=175,
                    facecolor=PANEL, edgecolor='#334455', linewidth=1.5, zorder=0)
    ax.add_patch(brain)
    ax.axvline(0, color='#334455', lw=0.8, alpha=0.5)
    for txt, x, y in [('L', -60, 78), ('R', 60, 78),
                       ('Anterior', 0, 83), ('Posterior', 0, -90)]:
        ax.text(x, y, txt, color='#556677',
                fontsize=8 if len(txt) > 1 else 11,
                fontweight='bold', ha='center', va='center')
    for u, v in draw_edges:
        both = u not in final_failed and v not in final_failed
        ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]],
                '-', color='#2a4a6a' if both else '#111122',
                lw=0.7, alpha=0.35 if both else 0.08, zorder=2)

    for ci, c in enumerate(unique_comms):
        members = [i for i in range(N) if labels_leiden[i] == c]
        ax.scatter([], [], c=comm_color[c], s=40, label=f'C{c}  (n={len(members)})',
                   edgecolors='white', linewidths=0.4)
    ax.legend(loc='lower left', fontsize=7, facecolor='#0d0d1a',
              edgecolor='#334455', labelcolor='white', framealpha=0.85,
              markerscale=1.0, handletextpad=0.4, borderpad=0.5)

    colors = [base_colors[i] if i not in final_failed else '#111122' for i in range(N)]
    sizes  = [55 if i not in final_failed else 20 for i in range(N)]
    ec     = ['white' if i not in final_failed else '#222233' for i in range(N)]
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=colors, edgecolors=ec,
               linewidths=0.5, zorder=4)
    ax.scatter([xy[attack, 0]], [xy[attack, 1]], s=200, marker='x',
               c=accent_color, linewidths=2, zorder=7)

    ax.text(0, 88,
            f'G = {G_metric:.2f}  |  {len(final_failed)} regions failed  |  {N - len(final_failed)} survived',
            ha='center', va='center', fontsize=8.5, color='#ddeeff', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=1.5, foreground='black')], zorder=11)

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, f'{tag}_result.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {out}')


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION 2 — AD Braak-staged cascade
# ══════════════════════════════════════════════════════════════════════════════
def make_ad_mp4(alpha=0.1):
    print(f'\n=== AD BRAAK STAGED CASCADE  α={alpha} ===')
    ad_history = run_ad_staged(alpha)

    stage_snapshots = []
    cum_before = frozenset()
    for entry in ad_history:
        stage_snapshots.append(dict(
            label     = entry['stage_label'],
            before    = cum_before,
            forced    = entry['forced'],
            c_rounds  = entry['cascade_rounds'],
            all_after = entry['all_removed'],
            G_val     = entry['G_val'],
        ))
        cum_before = entry['all_removed']

    final_all = stage_snapshots[-1]['all_after']
    G_final   = stage_snapshots[-1]['G_val']
    survived  = [i for i in range(N) if i not in final_all]

    # Frame schedule:
    # INTRO (healthy brain hold) → for each stage: force_pulse → force_hold →
    #   [cascade rounds] → stage_end → FREEZE result
    frame_data = []
    for f in range(HOLD):                            # intro: healthy brain
        frame_data.append(('intro', f))
    for si, snap in enumerate(stage_snapshots):
        for f in range(TRANS):
            frame_data.append(('force_pulse', si, f))
        for f in range(HOLD):
            frame_data.append(('force_hold', si, f))
        for ci in range(len(snap['c_rounds'])):
            for f in range(TRANS):
                frame_data.append(('casc_pulse', si, ci, f))
            for f in range(HOLD // 2):
                frame_data.append(('casc_hold', si, ci, f))
        for f in range(HOLD):
            frame_data.append(('stage_end', si, f))
    for f in range(FREEZE):
        frame_data.append(('result', f))

    # ── Figure: header strip + brain ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 11), facecolor=BG)
    fig.subplots_adjust(top=0.82, bottom=0.04, left=0.02, right=0.98)
    ax.set_facecolor(BG); ax.axis('off'); ax.set_aspect('equal')
    ax.set_xlim(-90, 90); ax.set_ylim(-98, 95)

    axh = fig.add_axes([0.02, 0.84, 0.96, 0.15])
    axh.set_facecolor('#0d0d1a'); axh.axis('off')
    axh.set_xlim(0, 1); axh.set_ylim(0, 1)

    # Static header title
    axh.text(0.02, 0.90, 'ALZHEIMER\'S DISEASE PROGRESSION', ha='left', va='top',
             color='#f39c12', fontsize=10, fontweight='bold')
    axh.text(0.02, 0.62, 'Model: Motter-Lai cascade failure  |  Braak staging order  |  α = ' + str(alpha),
             ha='left', va='top', color='#7799bb', fontsize=7.5)

    # Dynamic: current stage badge (orange box)
    stage_box = FancyBboxPatch((0.02, 0.04), 0.55, 0.40,
                                boxstyle='round,pad=0.02',
                                facecolor='#f39c12' + '22', edgecolor='#f39c12',
                                lw=1.5, zorder=2)
    axh.add_patch(stage_box)
    stage_lbl = axh.text(0.295, 0.24, 'Healthy Brain  —  Initial State',
                         ha='center', va='center', color='white',
                         fontsize=10, fontweight='bold')

    # Dynamic: what's being removed (right side of header)
    removed_lbl = axh.text(0.60, 0.82, '',
                           ha='left', va='top', color='#e74c3c',
                           fontsize=7.5, fontweight='bold')
    removed_body = axh.text(0.60, 0.62, '',
                            ha='left', va='top', color='#ccddee',
                            fontsize=7, linespacing=1.5)

    # G indicator in header top-right
    g_hdr = axh.text(0.98, 0.90, f'G = 1.000',
                     ha='right', va='top', color='#27ae60',
                     fontsize=10, fontweight='bold')

    # Brain outline
    brain = Ellipse((0, -10), width=150, height=175,
                    facecolor=PANEL, edgecolor='#334455', linewidth=1.5, zorder=0)
    ax.add_patch(brain)
    ax.axvline(0, color='#334455', lw=0.8, alpha=0.5, zorder=1)
    for txt, x, y in [('L', -60, 81), ('R', 60, 81),
                       ('Anterior', 0, 86), ('Posterior', 0, -93)]:
        ax.text(x, y, txt, color='#556677',
                fontsize=8 if len(txt) > 1 else 12,
                fontweight='bold', ha='center', va='center')

    for u, v in draw_edges:
        ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]],
                '-', color='#2a4a6a', lw=0.7, alpha=0.35, zorder=2)

    glow = ax.scatter(xy[:, 0], xy[:, 1], s=200, c=base_colors,
                      alpha=0.12, edgecolors='none', zorder=3)
    scat = ax.scatter(xy[:, 0], xy[:, 1], s=50, c=base_colors,
                      edgecolors='white', linewidths=0.5, zorder=4)

    bx, by, bw, bh = -60, -90, 120, 6
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                 boxstyle='round,pad=0.4',
                                 facecolor='#1a1a2a', edgecolor='#334455',
                                 lw=1, zorder=8))
    g_fill = FancyBboxPatch((bx, by), bw, bh, boxstyle='round,pad=0.4',
                              facecolor='#27ae60', edgecolor='none', zorder=9)
    ax.add_patch(g_fill)
    g_txt = ax.text(0, by + bh / 2, f'G = 1.000  ({N}/{N} alive)',
                    ha='center', va='center', color='white',
                    fontsize=9, fontweight='bold', zorder=10,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    # ── Helper ────────────────────────────────────────────────────────────────
    def get_cum_cascade(si, ci):
        cum = set(stage_snapshots[si]['before']) | set(stage_snapshots[si]['forced'])
        for k in range(ci + 1):
            cum |= stage_snapshots[si]['c_rounds'][k]
        return frozenset(cum)

    def set_header(stage_text, removed_header, removed_names, G_val):
        stage_lbl.set_text(stage_text)
        removed_lbl.set_text(removed_header)
        shown = removed_names[:10]
        extra = len(removed_names) - 10
        body = '\n'.join(f'  • {n}' for n in shown)
        if extra > 0:
            body += f'\n  … +{extra} more'
        removed_body.set_text(body)
        col = '#27ae60' if G_val > 0.75 else '#f39c12' if G_val > 0.5 else '#e74c3c'
        g_hdr.set_text(f'G = {G_val:.3f}')
        g_hdr.set_color(col)

    def region_names(node_set):
        return sorted(str(sc_ctx_labels[n]).replace('L_', 'L ').replace('R_', 'R ')
                      for n in node_set)

    # ── Animation update ───────────────────────────────────────────────────────
    def update(fi):
        fd = frame_data[fi]
        tag = fd[0]

        if tag == 'intro':
            update_scatter(scat, glow, frozenset())
            update_gbar(g_fill, g_txt, frozenset(), bx, bw)
            set_header('Healthy Brain  —  Initial State',
                       'No regions removed', [], 1.0)

        elif tag == 'force_pulse':
            _, si, f = fd
            snap = stage_snapshots[si]
            update_scatter(scat, glow, snap['before'],
                           forced=snap['forced'], sub_f=f)
            update_gbar(g_fill, g_txt, snap['before'], bx, bw)
            set_header(snap['label'].replace('\n', '  '),
                       f'Removing {len(snap["forced"])} regions:',
                       region_names(snap['forced']),
                       (N - len(snap['before'])) / N)

        elif tag == 'force_hold':
            _, si, f = fd
            snap = stage_snapshots[si]
            failed_now = snap['before'] | snap['forced']
            update_scatter(scat, glow, failed_now)
            update_gbar(g_fill, g_txt, failed_now, bx, bw)
            G_val = (N - len(failed_now)) / N
            set_header(snap['label'].replace('\n', '  '),
                       f'{len(snap["forced"])} regions degenerated:',
                       region_names(snap['forced']), G_val)

        elif tag == 'casc_pulse':
            _, si, ci, f = fd
            snap = stage_snapshots[si]
            failed_now = get_cum_cascade(si, ci - 1) if ci > 0 else \
                         snap['before'] | snap['forced']
            pulsing = snap['c_rounds'][ci]
            update_scatter(scat, glow, failed_now, pulsing=pulsing, sub_f=f)
            update_gbar(g_fill, g_txt, failed_now, bx, bw)
            G_val = (N - len(failed_now)) / N
            set_header(snap['label'].replace('\n', '  ') + f'  — Cascade round {ci+1}',
                       f'+{len(pulsing)} overloaded:',
                       region_names(pulsing), G_val)

        elif tag == 'casc_hold':
            _, si, ci, f = fd
            snap = stage_snapshots[si]
            failed_now = get_cum_cascade(si, ci)
            update_scatter(scat, glow, failed_now)
            update_gbar(g_fill, g_txt, failed_now, bx, bw)
            G_val = (N - len(failed_now)) / N
            set_header(snap['label'].replace('\n', '  ') + f'  — Cascade round {ci+1} done',
                       f'{len(snap["c_rounds"][ci])} overloaded:',
                       region_names(snap['c_rounds'][ci]), G_val)

        elif tag == 'stage_end':
            _, si, f = fd
            snap = stage_snapshots[si]
            update_scatter(scat, glow, snap['all_after'])
            update_gbar(g_fill, g_txt, snap['all_after'], bx, bw,
                        label_override=f'G = {snap["G_val"]:.3f}'
                                       f'  ({N-len(snap["all_after"])}/{N} alive)')
            set_header(snap['label'].replace('\n', '  ') + '  — Stage complete',
                       '', [], snap['G_val'])

        elif tag == 'result':
            update_scatter(scat, glow, final_all)
            update_gbar(g_fill, g_txt, final_all, bx, bw,
                        accent_color='#f39c12',
                        label_override=f'Network Survival  G = {G_final:.2f}'
                                       f'  ({len(survived)}/{N} alive)')
            set_header(f'All Braak Stages Complete  —  G = {G_final:.2f}',
                       f'{N - len(survived)} regions failed', [], G_final)

        return [scat, glow, g_fill, g_txt, stage_lbl, removed_lbl, removed_body, g_hdr]

    anim = animation.FuncAnimation(fig, update, frames=len(frame_data),
                                    interval=1000 // FPS, blit=True)
    out = os.path.join(FIGS_DIR, 'hcp_ml_brain_ad.mp4')
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2000,
                                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    anim.save(out, writer=writer, dpi=130)
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    # Find L and R superiorparietal indices
    node_R = int(np.argmax(L0_uw_arr))                          # R_superiorparietal (L0=153)
    node_L = next(i for i, l in enumerate(sc_ctx_labels)
                  if 'L_superiorparietal' in str(l))            # L_superiorparietal (L0=89)

    # R_superiorparietal — bad tolerance (catastrophic)
    make_hub_mp4(alpha=0.10, tag='hcp_ml_brain_R_bad',
                 accent_color='#e74c3c', attack=node_R)
    # R_superiorparietal — good tolerance (mostly contained)
    make_hub_mp4(alpha=0.40, tag='hcp_ml_brain_R_good',
                 accent_color='#27ae60', attack=node_R)
    # L_superiorparietal — bad tolerance (partial cascade)
    make_hub_mp4(alpha=0.10, tag='hcp_ml_brain_L_bad',
                 accent_color='#f39c12', attack=node_L)
    # L_superiorparietal — good tolerance (fully contained)
    make_hub_mp4(alpha=0.40, tag='hcp_ml_brain_L_good',
                 accent_color='#4A90D9', attack=node_L)
    print('\nDone.')
