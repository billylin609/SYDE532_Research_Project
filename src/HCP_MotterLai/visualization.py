"""HCP Motter-Lai — combined visualization.

Section 1: Alpha sweep — cascade damage vs tolerance (L vs R superiorparietal)
Section 2: Brain MP4s — hub attack (bad/good α, L and R)
Section 3: Brain MP4 — AD Braak-staged cascade
"""
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np
import networkx as nx

import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

from HCP_MotterLai.model import load_sc_and_loads, motter_lai, precompute_L0
from HCP_Community.model import detect_leiden, load_full_sc

_DIR     = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)
np.random.seed(42)

# ── Load config ────────────────────────────────────────────────────────────────
with open(os.path.join(_DIR, 'config.yaml')) as f:
    CFG = yaml.safe_load(f)

BG    = '#0a0a12'
PANEL = '#0f1520'

# ── Shared data loading ────────────────────────────────────────────────────────
print('Loading HCP SC…')
(sc_ctx, sc_ctx_labels, N, A_bin, G_w, G_uw,
 L0_uw_arr, L0_w_arr, degree_arr) = load_sc_and_loads()

# Precompute L0 once — reused by every motter_lai call to avoid
# redundant full-graph betweenness computation (saves ~320 calls during sweep)
print('Precomputing initial loads…')
_L0_precomputed   = precompute_L0(sc_ctx, weighted=False)
_L0_precomputed_w = precompute_L0(sc_ctx, weighted=True)

def resolve_node(label):
    """Return node index for a label string or 'argmax_betweenness'."""
    if label == 'argmax_betweenness':
        return int(np.argmax(L0_uw_arr))
    matches = [i for i, l in enumerate(sc_ctx_labels) if str(l) == label]
    if not matches:
        raise ValueError(f'Node label not found in sc_ctx_labels: {label!r}')
    return matches[0]

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


def get_xy(labels):
    xy = []
    for lbl in labels:
        s = str(lbl).replace('L_', '').replace('R_', '').lower()
        is_right = str(lbl).startswith('R_')
        x, y = DK_MNI_LEFT.get(s, (0, 0))
        xy.append((-x if is_right else x, y))
    return np.array(xy, dtype=float)


xy           = get_xy(sc_ctx_labels)
short_labels = [str(l).replace('L_', '').replace('R_', '').lower() for l in sc_ctx_labels]

# ── Leiden community colours ───────────────────────────────────────────────────
print('Running Leiden…')
COMM_PALETTE = ['#E05C5C', '#4A90D9', '#27AE60', '#F39C12', '#8E44AD', '#1ABC9C']
A_full, _, n_full, _, _, _, _, _ = load_full_sc()
labels_leiden_82, Q_leiden = detect_leiden(A_full, n_full)
labels_leiden = labels_leiden_82[:N]
unique_comms  = np.unique(labels_leiden)
comm_color    = {c: COMM_PALETTE[i % len(COMM_PALETTE)] for i, c in enumerate(unique_comms)}
base_colors   = [comm_color[labels_leiden[i]] for i in range(N)]
print(f'  Leiden Q={Q_leiden:.4f}, {len(unique_comms)} communities')

# ── Edge setup — top 60% of non-zero weights ───────────────────────────────────
all_edges    = list(G_w.edges(data=True))
edge_wts     = np.array([d['weight'] for _, _, d in all_edges])
nonzero_wts  = edge_wts[edge_wts > 0]
edge_thresh  = np.percentile(nonzero_wts, 40)
w_min_e, w_max_e = edge_thresh, nonzero_wts.max()

draw_edges    = [(u, v) for u, v, d in all_edges if d['weight'] >= edge_thresh]
draw_edge_wt  = np.array([d['weight'] for u, v, d in all_edges if d['weight'] >= edge_thresh])
draw_edge_norm = (draw_edge_wt - w_min_e) / (w_max_e - w_min_e + 1e-9)
draw_edge_lw   = (0.3 + 2.2 * draw_edge_norm).tolist()
draw_edge_segs = [[[xy[u, 0], xy[u, 1]], [xy[v, 0], xy[v, 1]]] for u, v in draw_edges]


def make_edge_collection(failed_set):
    segs, lws = [], []
    for (u, v), seg, lw in zip(draw_edges, draw_edge_segs, draw_edge_lw):
        if u not in failed_set and v not in failed_set:
            segs.append(seg)
            lws.append(lw)
    return LineCollection(segs, colors='#2a4a6a', linewidths=lws, alpha=0.35, zorder=2), segs, lws


def refresh_edges(col, failed_set):
    segs, lws = [], []
    for (u, v), seg, lw in zip(draw_edges, draw_edge_segs, draw_edge_lw):
        if u not in failed_set and v not in failed_set:
            segs.append(seg)
            lws.append(lw)
    col.set_segments(segs)
    col.set_linewidth(lws)


# ── Shared scatter helpers ─────────────────────────────────────────────────────
FPS        = 15          # reduced from 24 — fewer frames, same visual duration
HOLD_SEC   = 1.0         # reduced from 2
TRANS_SEC  = 0.5         # reduced from 1
FREEZE_SEC = 1.5         # reduced from 2
HOLD   = max(1, int(FPS * HOLD_SEC))
TRANS  = max(1, int(FPS * TRANS_SEC))
FREEZE = max(1, int(FPS * FREEZE_SEC))


def update_scatter(scat, glow, failed_now, pulsing=frozenset(), forced=frozenset(), sub_f=0):
    colors, sizes, ec = [], [], []
    for i in range(N):
        if i in failed_now:
            colors.append('#111122'); sizes.append(18); ec.append('#222233')
        elif i in pulsing:
            pulse = 0.5 + 0.5 * np.sin(sub_f / FPS * 2 * np.pi * 2)
            colors.append((1.0, 0.2 * pulse, 0.05 * pulse, 1.0))
            sizes.append(80 * (1.3 + 0.3 * pulse)); ec.append('#ffff00')
        elif i in forced:
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
    g_fill.set_facecolor(accent_color or (
        '#27ae60' if G_val > 0.75 else '#f39c12' if G_val > 0.5 else '#e74c3c'))
    g_txt.set_text(label_override or f'G = {G_val:.3f}  ({N - len(failed_set)}/{N} alive)')


def brain_axes(fig, ax):
    ax.set_facecolor(BG); ax.axis('off'); ax.set_aspect('equal')
    ax.set_xlim(-90, 90); ax.set_ylim(-95, 95)
    ax.add_patch(Ellipse((0, -10), width=150, height=175,
                         facecolor=PANEL, edgecolor='#334455', linewidth=1.5, zorder=0))
    ax.axvline(0, color='#334455', lw=0.8, alpha=0.5, zorder=1)
    for txt, x, y in [('L', -60, 78), ('R', 60, 78),
                       ('Anterior', 0, 83), ('Posterior', 0, -90)]:
        ax.text(x, y, txt, color='#556677',
                fontsize=8 if len(txt) > 1 else 11,
                fontweight='bold', ha='center', va='center')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Alpha sweep: cascade damage vs tolerance
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== ALPHA SWEEP ===')
sw_cfg  = CFG['alpha_sweep']
ALPHAS  = np.linspace(sw_cfg['alpha_min'], sw_cfg['alpha_max'], sw_cfg['n_points'])
sw_nodes = sw_cfg['nodes']

SWEEP_COLORS  = ['#e74c3c', '#f39c12', '#4A90D9', '#27AE60', '#8E44AD', '#1ABC9C']
SWEEP_MARKERS = ['o', 's', '^', 'D', 'v', 'p']

def node_dir(label):
    """Return (and create) a per-node output subfolder."""
    d = os.path.join(FIGS_DIR, label)
    os.makedirs(d, exist_ok=True)
    return d


def crit_alpha(dmg, alphas):
    for a, d in zip(alphas, dmg):
        if d < 0.05:
            return a
    return alphas[-1]


# ── Parallel alpha sweep ───────────────────────────────────────────────────────
# Worker must be a module-level picklable function for ProcessPoolExecutor.
def _sweep_worker(args):
    """Run motter_lai for one (node_idx, alpha) pair. Top-level for pickling."""
    idx, alpha, A, L0_arr = args
    return motter_lai(A, idx, alpha, L0_arr=L0_arr)[0]


sweep_results = {}
node_index_map = {lbl: resolve_node(lbl) for lbl in sw_nodes}
tasks = [(lbl, idx, a) for lbl in sw_nodes for a in ALPHAS
         for idx in [node_index_map[lbl]]]

print(f'Sweeping {len(sw_nodes)} nodes × {len(ALPHAS)} alphas in parallel…')
worker_args = [(node_index_map[lbl], a, sc_ctx, _L0_precomputed)
               for lbl in sw_nodes for a in ALPHAS]

with ProcessPoolExecutor() as pool:
    all_G = list(pool.map(_sweep_worker, worker_args, chunksize=4))

# Reshape results back to per-node
n_alpha = len(ALPHAS)
for i, lbl in enumerate(sw_nodes):
    idx = node_index_map[lbl]
    sweep_results[lbl] = (idx, all_G[i * n_alpha:(i + 1) * n_alpha])
    print(f'  Done: {lbl}')

    # ── Per-node individual sweep plot ────────────────────────────────────────
    _, G_vals = sweep_results[lbl]
    dmg   = 1 - np.array(G_vals)
    crit  = crit_alpha(dmg, ALPHAS)
    color = SWEEP_COLORS[sw_nodes.index(lbl) % len(SWEEP_COLORS)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ALPHAS, dmg, 'o-', color=color, linewidth=2, markersize=4,
            label=f'{lbl}  (L₀={L0_uw_arr[idx]:.0f})')
    ax.axvline(crit, color=color, lw=1.5, ls=':', alpha=0.8,
               label=f'Critical α ≈ {crit:.2f}')
    for entry in CFG['hub_attacks']:
        if entry['label'] == lbl:
            for sc in entry['scenarios']:
                a = sc['alpha']
                d = dmg[np.argmin(np.abs(ALPHAS - a))]
                ax.scatter([a], [d], s=100, color=color, zorder=5,
                           edgecolors='black', lw=0.8)
                ax.annotate(f'α={a}', (a, d), xytext=(a + 0.02, d + 0.03),
                            fontsize=8, color=color, fontweight='bold')
    ax.axhline(0.0, color='black', lw=0.8, ls='--', alpha=0.4)
    ax.set_xlabel('Tolerance parameter  α', fontsize=11)
    ax.set_ylabel('Cascade damage  =  1 − G', fontsize=11)
    ax.set_title(f'Cascade Damage vs Tolerance — {lbl}  [Unweighted]',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(sw_cfg['alpha_min'], sw_cfg['alpha_max'])
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(node_dir(lbl), 'alpha_sweep.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTED vs UNWEIGHTED CASCADE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== WEIGHTED vs UNWEIGHTED SWEEP ===')
print('Running weighted sweep (uses inverse-weight distances for betweenness)…')

def _sweep_worker_w(args):
    idx, alpha, A, L0_arr = args
    return motter_lai(A, idx, alpha, weighted=True, L0_arr=L0_arr)[0]

worker_args_w = [(node_index_map[lbl], a, sc_ctx, _L0_precomputed_w)
                 for lbl in sw_nodes for a in ALPHAS]
with ProcessPoolExecutor() as pool:
    all_G_w = list(pool.map(_sweep_worker_w, worker_args_w, chunksize=4))

sweep_results_w = {}
for i, lbl in enumerate(sw_nodes):
    idx = node_index_map[lbl]
    sweep_results_w[lbl] = (idx, all_G_w[i * n_alpha:(i + 1) * n_alpha])

# ── Overlay plot: UW (transparent) + W (solid) on the same axes ──────────────
fig, ax = plt.subplots(figsize=(12, 5))
for i, lbl in enumerate(sw_nodes):
    color = SWEEP_COLORS[i % len(SWEEP_COLORS)]
    mark  = SWEEP_MARKERS[i % len(SWEEP_MARKERS)]

    _, G_uw = sweep_results[lbl]
    dmg_uw  = 1 - np.array(G_uw)
    _, G_w  = sweep_results_w[lbl]
    dmg_w   = 1 - np.array(G_w)

    # Unweighted — dashed, full opacity
    ax.plot(ALPHAS, dmg_uw, f'{mark}--', color=color, linewidth=1.4,
            markersize=2.5, alpha=1.0)
    # Weighted — solid, full opacity, carries the legend entry
    crit_w = crit_alpha(dmg_w, ALPHAS)
    ax.plot(ALPHAS, dmg_w, f'{mark}-', color=color, linewidth=2.0,
            markersize=3, alpha=1.0,
            label=f'{lbl}  W crit α≈{crit_w:.2f}')
    ax.axvline(crit_w, color=color, lw=0.8, ls=':', alpha=0.5)

# Invisible proxy lines for the UW / W legend entries
from matplotlib.lines import Line2D
ax.add_line(Line2D([], [], color='grey', lw=1.5, ls='--',
                   label='── Unweighted (dashed)'))
ax.add_line(Line2D([], [], color='grey', lw=1.5, ls='-',
                   label='── Weighted (solid)'))

ax.axhline(0.0, color='black', lw=0.8, ls='--', alpha=0.4)
ax.set_xlabel('Tolerance parameter  α', fontsize=12)
ax.set_ylabel('Cascade damage  =  1 − G', fontsize=12)
ax.set_title('Cascade Damage vs Tolerance — Weighted vs Unweighted  (all nodes)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=7.5, ncol=2, framealpha=0.85)
ax.set_xlim(sw_cfg['alpha_min'], sw_cfg['alpha_max'])
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(FIGS_DIR, 'hcp_ml_alpha_sweep_compare.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# ── Comparison table ──────────────────────────────────────────────────────────
# Report damage (1-G) at 3 representative alpha values
report_alphas = [0.1, 0.25, 0.4, 0.6, 1.0]
ra_idx = [int(np.argmin(np.abs(ALPHAS - a))) for a in report_alphas]

print()
print('=' * 105)
print('CASCADE DAMAGE (1 - G)  —  Unweighted (UW) vs Weighted (W) betweenness')
print('=' * 105)
header = f'  {"Node":<28}' + ''.join(
    f'  α={a:.2f} UW    W   Δ' for a in report_alphas)
print(header)
print('  ' + '-' * 103)

for lbl in sw_nodes:
    _, G_uw_vals = sweep_results[lbl]
    _, G_w_vals  = sweep_results_w[lbl]
    dmg_uw = 1 - np.array(G_uw_vals)
    dmg_w  = 1 - np.array(G_w_vals)
    row = f'  {lbl:<28}'
    for ai in ra_idx:
        du = dmg_uw[ai]; dw = dmg_w[ai]
        row += f'  {du:5.3f} {dw:5.3f} {dw-du:+.3f}'
    print(row)

print()
print('  Δ = W − UW  (positive = weighted network is more fragile at that α)')

# ── Critical alpha comparison ─────────────────────────────────────────────────
print()
print('=' * 65)
print('CRITICAL α  (first α where damage < 0.05)')
print('=' * 65)
print(f'  {"Node":<28}  {"UW crit α":>10}  {"W crit α":>10}  {"Δ":>8}')
print('  ' + '-' * 63)
for lbl in sw_nodes:
    _, G_uw_vals = sweep_results[lbl]
    _, G_w_vals  = sweep_results_w[lbl]
    ca_uw = crit_alpha(1 - np.array(G_uw_vals), ALPHAS)
    ca_w  = crit_alpha(1 - np.array(G_w_vals),  ALPHAS)
    print(f'  {lbl:<28}  {ca_uw:>10.3f}  {ca_w:>10.3f}  {ca_w - ca_uw:>+8.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Hub attack brain MP4s
# ══════════════════════════════════════════════════════════════════════════════
def make_hub_mp4(alpha, tag, accent_color, attack=None, out_dir=None):
    if attack is None:
        attack = resolve_node('argmax_betweenness')
    attack_name = str(sc_ctx_labels[attack])
    if out_dir is None:
        out_dir = node_dir(attack_name)

    print(f'\n=== HUB ATTACK  α={alpha}  ({tag}) ===')
    G_metric, n_failed, history = motter_lai(sc_ctx, attack, alpha,
                                              return_history=True,
                                              L0_arr=_L0_precomputed)
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

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
    fig.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98)
    brain_axes(fig, ax)

    fig.suptitle(
        f'Hub Attack — {attack_name}  |  α = {alpha}  [Unweighted]  '
        f'({"cascade failure" if G_metric < 0.85 else "contained"})',
        color=accent_color, fontsize=11, fontweight='bold', y=0.97)

    edge_col, _, _ = make_edge_collection(frozenset())
    ax.add_collection(edge_col)

    for ci, c in enumerate(unique_comms):
        members = [i for i in range(N) if labels_leiden[i] == c]
        ax.scatter([], [], c=comm_color[c], s=40, label=f'C{c}  (n={len(members)})',
                   edgecolors='white', linewidths=0.4)
    ax.legend(loc='lower left', fontsize=7, facecolor='#0d0d1a',
              edgecolor='#334455', labelcolor='white', framealpha=0.85,
              markerscale=1.0, handletextpad=0.4, borderpad=0.5)

    glow = ax.scatter(xy[:, 0], xy[:, 1], s=220, c=base_colors,
                      alpha=0.12, edgecolors='none', zorder=3)
    scat = ax.scatter(xy[:, 0], xy[:, 1], s=55, c=base_colors,
                      edgecolors='white', linewidths=0.5, zorder=4)

    ax.scatter([xy[attack, 0]], [xy[attack, 1]], s=340, marker='*',
               c=accent_color, edgecolors='#ffff00', linewidths=1.0, zorder=7)
    ax.annotate(attack_name.replace('R_', 'R '),
                xy=(xy[attack, 0], xy[attack, 1]),
                xytext=(xy[attack, 0] + 12, xy[attack, 1] + 9),
                color=accent_color, fontsize=7, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')],
                arrowprops=dict(arrowstyle='->', color=accent_color, lw=1.0), zorder=8)

    rnd_txt = ax.text(0, 88, 'Round 0  —  Seed removed',
                      ha='center', va='center', fontsize=8.5,
                      color='#f39c12', fontweight='bold', zorder=11,
                      path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    def update(fi):
        mode, ri, sub_f, is_trans = frame_data[fi]
        failed_now  = cumulative[ri]
        next_failed = cumulative[ri + 1] if ri < len(history) - 1 else failed_now
        pulsing     = (next_failed - failed_now) if is_trans else frozenset()
        update_scatter(scat, glow, failed_now, pulsing=pulsing, sub_f=sub_f)
        refresh_edges(edge_col, failed_now if mode == 'cascade' else final_failed)
        if mode == 'cascade':
            rnd_txt.set_text(
                f'Round 0  —  Seed removed  ({len(failed_now)} failed)' if ri == 0 else
                f'Round {ri}  —  +{len(history[ri])} overloaded  ({len(failed_now)}/{N} failed)')
        else:
            update_scatter(scat, glow, final_failed)
            rnd_txt.set_text(f'Cascade complete — G = {G_metric:.2f}  ({len(survived)}/{N} survived)')
        return [scat, glow, rnd_txt, edge_col]

    anim = animation.FuncAnimation(fig, update, frames=len(frame_data),
                                    interval=1000 // FPS, blit=True)
    out = os.path.join(out_dir, f'{tag}.mp4')
    anim.save(out, writer=animation.FFMpegWriter(
        fps=FPS, bitrate=1500,
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                    '-preset', 'fast', '-crf', '23']), dpi=110)
    plt.close(fig)
    print(f'Saved: {out}')
    # Return simulation data so the caller can build the combined 2×2 result PNG
    return dict(attack=attack, final_failed=final_failed, G_metric=G_metric,
                alpha=alpha, color=accent_color, tag=tag, out_dir=out_dir)


def _draw_brain_state(ax, attack, failed_set, accent_color, title):
    """Draw one brain panel (start or end state) into ax."""
    brain_axes(None, ax)

    # Edges — alive edges full colour, dead edges faded
    for (u, v), seg, lw in zip(draw_edges, draw_edge_segs, draw_edge_lw):
        both = u not in failed_set and v not in failed_set
        xs = [seg[0][0], seg[1][0]]; ys = [seg[0][1], seg[1][1]]
        ax.plot(xs, ys, '-',
                color='#2a4a6a' if both else '#111122',
                lw=lw if both else 0.3,
                alpha=0.35 if both else 0.08, zorder=2)

    colors = [base_colors[i] if i not in failed_set else '#111122' for i in range(N)]
    sizes  = [55 if i not in failed_set else 18 for i in range(N)]
    ec     = ['white' if i not in failed_set else '#222233' for i in range(N)]
    ax.scatter(xy[:, 0], xy[:, 1], s=[s * 3.5 for s in sizes],
               c=colors, alpha=0.12, edgecolors='none', zorder=3)
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=colors,
               edgecolors=ec, linewidths=0.5, zorder=4)

    # Attack node marker
    ax.scatter([xy[attack, 0]], [xy[attack, 1]], s=260, marker='*',
               c=accent_color, edgecolors='#ffff00', linewidths=0.8, zorder=7)

    ax.set_title(title, color=accent_color, fontsize=9, fontweight='bold', pad=5)


def save_combined_result_png(attack, good, bad, out_dir):
    """
    2×2 result figure per hub-attack node.
      Row 0 (top)    — good scenario (high α, contained)
      Row 1 (bottom) — bad  scenario (low α, cascade failure)
      Col 0 (left)   — initial state (all alive, attack node marked)
      Col 1 (right)  — final state   (failed nodes dimmed)
    """
    attack_name = str(sc_ctx_labels[attack])
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor=BG)
    fig.suptitle(f'Hub Attack — {attack_name}  |  Cascade Outcome Comparison',
                 color='white', fontsize=13, fontweight='bold', y=1.005)

    empty_set = frozenset()
    rows = [
        ('GOOD', good['alpha'], good['color'], good['final_failed'], good['G_metric']),
        ('BAD',  bad['alpha'],  bad['color'],  bad['final_failed'],  bad['G_metric']),
    ]
    for row_i, (label, alpha, color, final_failed, G_metric) in enumerate(rows):
        survived = N - len(final_failed)
        outcome  = 'Contained' if G_metric >= 0.85 else 'Cascade Failure'

        _draw_brain_state(axes[row_i, 0], attack, empty_set, color,
                          f'{label}  α={alpha}  — Initial state  (all {N} alive)')
        _draw_brain_state(axes[row_i, 1], attack, final_failed, color,
                          f'{label}  α={alpha}  — Final state  '
                          f'G={G_metric:.2f}  {survived}/{N} survived  [{outcome}]')

    # Row labels on the left edge
    for row_i, txt in enumerate(['GOOD (high α)', 'BAD (low α)']):
        fig.text(0.01, 0.75 - row_i * 0.5, txt,
                 va='center', ha='left', rotation=90,
                 color='white', fontsize=11, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(out_dir, 'result_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {out}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AD Braak-staged cascade MP4
# ══════════════════════════════════════════════════════════════════════════════
def nodes_matching(keywords):
    return [i for i, s in enumerate(short_labels) if any(k in s for k in keywords)]


BRAAK_STAGES = [
    ('Braak I–II\n(Transentorhinal)',  nodes_matching(['entorhinal'])),
    ('Braak III–IV\n(Limbic)',         nodes_matching(['parahippocampal', 'isthmuscingulate',
                                                        'posteriorcingulate'])),
    ('Braak V\n(Early Neocortical)',   nodes_matching(['inferiortemporal', 'middletemporal',
                                                        'fusiform', 'lingual', 'pericalcarine',
                                                        'lateraloccipital'])),
    ('Braak VI\n(Full Neocortical)',   nodes_matching(['superiorparietal', 'precuneus',
                                                        'superiorfrontal', 'caudalmiddlefrontal',
                                                        'superiortemporal'])),
]


def run_ad_staged(alpha=0.1):
    C = {i: (1 + alpha) * float(_L0_precomputed[i]) for i in range(N)}
    all_removed = set()
    history = []
    for stage_label, forced_nodes in BRAAK_STAGES:
        forced = frozenset(n for n in forced_nodes if n not in all_removed)
        all_removed |= forced
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
            lcc   = max(nx.connected_components(G_fin), key=len) if G_fin.number_of_nodes() > 0 else set()
            G_val = len(lcc) / N
        else:
            G_val = 0.0
        history.append(dict(stage_label=stage_label, forced=forced,
                            cascade_rounds=cascade_rounds,
                            all_removed=frozenset(all_removed), G_val=G_val))
        print(f'  {stage_label.split(chr(10))[0]}: forced={len(forced)}, '
              f'cascade={sum(len(r) for r in cascade_rounds)}, G={G_val:.3f}')
    return history


def make_ad_mp4(alpha=0.1):
    print(f'\n=== AD BRAAK STAGED CASCADE  α={alpha} ===')
    ad_history = run_ad_staged(alpha)

    stage_snapshots = []
    cum_before = frozenset()
    for entry in ad_history:
        stage_snapshots.append(dict(
            label=entry['stage_label'], before=cum_before,
            forced=entry['forced'], c_rounds=entry['cascade_rounds'],
            all_after=entry['all_removed'], G_val=entry['G_val'],
        ))
        cum_before = entry['all_removed']

    final_all = stage_snapshots[-1]['all_after']
    G_final   = stage_snapshots[-1]['G_val']
    survived  = [i for i in range(N) if i not in final_all]

    frame_data = []
    for f in range(HOLD):
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

    fig, ax = plt.subplots(figsize=(10, 11), facecolor=BG)
    fig.subplots_adjust(top=0.82, bottom=0.04, left=0.02, right=0.98)
    brain_axes(fig, ax)
    ax.set_ylim(-98, 95)

    axh = fig.add_axes([0.02, 0.84, 0.96, 0.15])
    axh.set_facecolor('#0d0d1a'); axh.axis('off')
    axh.set_xlim(0, 1); axh.set_ylim(0, 1)
    axh.text(0.02, 0.90, "ALZHEIMER'S DISEASE PROGRESSION",
             ha='left', va='top', color='#f39c12', fontsize=10, fontweight='bold')
    axh.text(0.02, 0.62,
             f'Model: Motter-Lai cascade failure  |  Braak staging order  |  α = {alpha}',
             ha='left', va='top', color='#7799bb', fontsize=7.5)

    stage_box = FancyBboxPatch((0.02, 0.04), 0.55, 0.40, boxstyle='round,pad=0.02',
                                facecolor='#f39c12' + '22', edgecolor='#f39c12',
                                lw=1.5, zorder=2)
    axh.add_patch(stage_box)
    stage_lbl    = axh.text(0.295, 0.24, 'Healthy Brain  —  Initial State',
                            ha='center', va='center', color='white',
                            fontsize=10, fontweight='bold')
    removed_lbl  = axh.text(0.60, 0.82, '', ha='left', va='top',
                            color='#e74c3c', fontsize=7.5, fontweight='bold')
    removed_body = axh.text(0.60, 0.62, '', ha='left', va='top',
                            color='#ccddee', fontsize=7, linespacing=1.5)
    g_hdr        = axh.text(0.98, 0.90, 'G = 1.000', ha='right', va='top',
                            color='#27ae60', fontsize=10, fontweight='bold')

    edge_col_ad, _, _ = make_edge_collection(frozenset())
    ax.add_collection(edge_col_ad)

    glow = ax.scatter(xy[:, 0], xy[:, 1], s=200, c=base_colors,
                      alpha=0.12, edgecolors='none', zorder=3)
    scat = ax.scatter(xy[:, 0], xy[:, 1], s=50, c=base_colors,
                      edgecolors='white', linewidths=0.5, zorder=4)

    bx, by, bw, bh = -60, -90, 120, 6
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle='round,pad=0.4',
                                facecolor='#1a1a2a', edgecolor='#334455', lw=1, zorder=8))
    g_fill = FancyBboxPatch((bx, by), bw, bh, boxstyle='round,pad=0.4',
                             facecolor='#27ae60', edgecolor='none', zorder=9)
    ax.add_patch(g_fill)
    g_txt = ax.text(0, by + bh / 2, f'G = 1.000  ({N}/{N} alive)',
                    ha='center', va='center', color='white',
                    fontsize=9, fontweight='bold', zorder=10,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

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
        body  = '\n'.join(f'  • {n}' for n in shown)
        if extra > 0:
            body += f'\n  … +{extra} more'
        removed_body.set_text(body)
        col = '#27ae60' if G_val > 0.75 else '#f39c12' if G_val > 0.5 else '#e74c3c'
        g_hdr.set_text(f'G = {G_val:.3f}')
        g_hdr.set_color(col)

    def region_names(node_set):
        return sorted(str(sc_ctx_labels[n]).replace('L_', 'L ').replace('R_', 'R ')
                      for n in node_set)

    def update(fi):
        fd  = frame_data[fi]
        tag = fd[0]

        if tag == 'intro':
            update_scatter(scat, glow, frozenset())
            refresh_edges(edge_col_ad, frozenset())
            update_gbar(g_fill, g_txt, frozenset(), bx, bw)
            set_header('Healthy Brain  —  Initial State', 'No regions removed', [], 1.0)

        elif tag == 'force_pulse':
            _, si, f = fd
            snap = stage_snapshots[si]
            update_scatter(scat, glow, snap['before'], forced=snap['forced'], sub_f=f)
            refresh_edges(edge_col_ad, snap['before'])
            update_gbar(g_fill, g_txt, snap['before'], bx, bw)
            set_header(snap['label'].replace('\n', '  '),
                       f'Removing {len(snap["forced"])} regions:',
                       region_names(snap['forced']), (N - len(snap['before'])) / N)

        elif tag == 'force_hold':
            _, si, f = fd
            snap = stage_snapshots[si]
            failed_now = snap['before'] | snap['forced']
            update_scatter(scat, glow, failed_now)
            refresh_edges(edge_col_ad, failed_now)
            update_gbar(g_fill, g_txt, failed_now, bx, bw)
            set_header(snap['label'].replace('\n', '  '),
                       f'{len(snap["forced"])} regions degenerated:',
                       region_names(snap['forced']), (N - len(failed_now)) / N)

        elif tag == 'casc_pulse':
            _, si, ci, f = fd
            snap = stage_snapshots[si]
            failed_now = get_cum_cascade(si, ci - 1) if ci > 0 else snap['before'] | snap['forced']
            pulsing = snap['c_rounds'][ci]
            update_scatter(scat, glow, failed_now, pulsing=pulsing, sub_f=f)
            refresh_edges(edge_col_ad, failed_now)
            update_gbar(g_fill, g_txt, failed_now, bx, bw)
            set_header(snap['label'].replace('\n', '  ') + f'  — Cascade round {ci+1}',
                       f'+{len(pulsing)} overloaded:', region_names(pulsing),
                       (N - len(failed_now)) / N)

        elif tag == 'casc_hold':
            _, si, ci, f = fd
            snap = stage_snapshots[si]
            failed_now = get_cum_cascade(si, ci)
            update_scatter(scat, glow, failed_now)
            refresh_edges(edge_col_ad, failed_now)
            update_gbar(g_fill, g_txt, failed_now, bx, bw)
            set_header(snap['label'].replace('\n', '  ') + f'  — Cascade round {ci+1} done',
                       f'{len(snap["c_rounds"][ci])} overloaded:',
                       region_names(snap['c_rounds'][ci]), (N - len(failed_now)) / N)

        elif tag == 'stage_end':
            _, si, f = fd
            snap = stage_snapshots[si]
            update_scatter(scat, glow, snap['all_after'])
            refresh_edges(edge_col_ad, snap['all_after'])
            update_gbar(g_fill, g_txt, snap['all_after'], bx, bw,
                        label_override=f'G = {snap["G_val"]:.3f}'
                                       f'  ({N-len(snap["all_after"])}/{N} alive)')
            set_header(snap['label'].replace('\n', '  ') + '  — Stage complete',
                       '', [], snap['G_val'])

        elif tag == 'result':
            update_scatter(scat, glow, final_all)
            refresh_edges(edge_col_ad, final_all)
            update_gbar(g_fill, g_txt, final_all, bx, bw, accent_color='#f39c12',
                        label_override=f'Network Survival  G = {G_final:.2f}'
                                       f'  ({len(survived)}/{N} alive)')
            set_header(f'All Braak Stages Complete  —  G = {G_final:.2f}',
                       f'{N - len(survived)} regions failed', [], G_final)

        return [scat, glow, g_fill, g_txt, stage_lbl,
                removed_lbl, removed_body, g_hdr, edge_col_ad]

    anim = animation.FuncAnimation(fig, update, frames=len(frame_data),
                                    interval=1000 // FPS, blit=True)
    out = os.path.join(FIGS_DIR, 'hcp_ml_brain_ad.mp4')
    anim.save(out, writer=animation.FFMpegWriter(
        fps=FPS, bitrate=1500,
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                    '-preset', 'fast', '-crf', '23']), dpi=110)
    plt.close(fig)
    print(f'Saved: {out}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    for entry in CFG['hub_attacks']:
        attack_idx   = resolve_node(entry['label'])
        out_d        = node_dir(entry['label'])
        scenario_res = []
        for sc in entry['scenarios']:
            res = make_hub_mp4(alpha=sc['alpha'], tag=sc['tag'],
                               accent_color=sc['color'], attack=attack_idx,
                               out_dir=out_d)
            scenario_res.append(res)

        # Sort by G_metric: higher G = good (contained), lower G = bad (cascade)
        scenario_res.sort(key=lambda r: r['G_metric'], reverse=True)
        good_res, bad_res = scenario_res[0], scenario_res[-1]
        save_combined_result_png(attack_idx, good_res, bad_res, out_d)

    make_ad_mp4(alpha=CFG['ad_cascade']['alpha'])

    # ── Centrality-G correlation analysis ─────────────────────────────────────
    from scipy.stats import pearsonr, spearmanr
    from HCP_Centrality.model import (
        load_sc_graph, compute_degree_centrality, compute_betweenness_centrality,
        compute_closeness_centrality, compute_eigenvector_centrality,
    )

    print('\n=== CENTRALITY–CASCADE VULNERABILITY CORRELATION ===')
    print('Computing centrality measures from HCP_Centrality…')

    sc_c, lbl_c, n_c, G_c = load_sc_graph()
    deg_arr  = compute_degree_centrality(G_c, n_c)
    btw_arr  = compute_betweenness_centrality(G_c, n_c)
    cls_arr  = compute_closeness_centrality(G_c, n_c)
    eig_arr  = compute_eigenvector_centrality(G_c, n_c)

    centralities = [
        ('Degree',      deg_arr),
        ('Betweenness', btw_arr),
        ('Closeness',   cls_arr),
        ('Eigenvector', eig_arr),
    ]

    CORR_ALPHAS = [0.10, 0.25, 0.40]
    print(f'Computing G for all {n_c} nodes at α ∈ {CORR_ALPHAS} (parallelised)…')

    def _g_worker(args):
        node_i, alpha, A, L0 = args
        return motter_lai(A, node_i, alpha, L0_arr=L0)[0]

    corr_lines = []
    G_by_alpha = {}
    for alpha in CORR_ALPHAS:
        args = [(i, alpha, sc_ctx, _L0_precomputed) for i in range(n_c)]
        with ProcessPoolExecutor() as pool:
            G_vals = np.array(list(pool.map(_g_worker, args, chunksize=8)))
        G_by_alpha[alpha] = G_vals

    # ── Build report ──────────────────────────────────────────────────────────
    sep   = '=' * 82
    lines = [sep,
             'CENTRALITY–CASCADE VULNERABILITY CORRELATION',
             'Correlating each node\'s centrality score with its cascade G  '
             '(G = surviving fraction of LCC)',
             'Pearson r measures linear association; Spearman ρ is rank-based '
             '(robust to outliers).',
             'A negative correlation means higher centrality → more damage when that node fails.',
             sep, '']

    for alpha in CORR_ALPHAS:
        G_arr = G_by_alpha[alpha]
        dmg   = 1 - G_arr
        lines.append(f'α = {alpha}  (mean G = {G_arr.mean():.3f}, '
                     f'mean damage = {dmg.mean():.3f})')
        lines.append('-' * 60)
        lines.append(f'  {"Centrality":<14} {"Pearson r":>10} {"p-value":>10} '
                     f'{"Spearman ρ":>12} {"p-value":>10}  Interpretation')
        lines.append('  ' + '-' * 74)
        for name, c_arr in centralities:
            r_p, p_p = pearsonr(c_arr, dmg)
            r_s, p_s = spearmanr(c_arr, dmg)
            sig_p = '***' if p_p < 0.001 else ('**' if p_p < 0.01 else ('*' if p_p < 0.05 else ''))
            sig_s = '***' if p_s < 0.001 else ('**' if p_s < 0.01 else ('*' if p_s < 0.05 else ''))
            interp = ('strong predictor' if abs(r_s) > 0.5 else
                      'moderate predictor' if abs(r_s) > 0.3 else 'weak predictor')
            direction = '(higher → more damage)' if r_s > 0 else '(higher → less damage)'
            lines.append(f'  {name:<14} {r_p:>+10.4f} {p_p:>9.4f}{sig_p:<3} '
                         f'{r_s:>+12.4f} {p_s:>9.4f}{sig_s:<3}  {interp} {direction}')
        lines.append('')

    # ── Best predictor summary ─────────────────────────────────────────────────
    lines.append(sep)
    lines.append('BEST PREDICTOR PER ALPHA  (by |Spearman ρ|)')
    lines.append('-' * 60)
    for alpha in CORR_ALPHAS:
        G_arr = G_by_alpha[alpha]
        dmg   = 1 - G_arr
        best_name, best_r = max(
            ((name, spearmanr(c_arr, dmg).statistic) for name, c_arr in centralities),
            key=lambda x: abs(x[1]))
        lines.append(f'  α={alpha}:  {best_name:<14}  ρ = {best_r:+.4f}')
    lines.append('')

    # ── Per-node top-10 most vulnerable ───────────────────────────────────────
    for alpha in CORR_ALPHAS:
        G_arr = G_by_alpha[alpha]
        dmg   = 1 - G_arr
        top10 = np.argsort(dmg)[-10:][::-1]
        lines.append(f'TOP-10 MOST VULNERABLE NODES  (α={alpha}, highest damage)')
        lines.append('-' * 60)
        lines.append(f'  {"Rank":<5} {"Region":<28} {"Damage":>8}  {"Degree":>8} '
                     f'{"Betweenness":>12} {"Closeness":>10} {"Eigenvector":>12}')
        lines.append('  ' + '-' * 80)
        for rank, i in enumerate(top10, 1):
            lines.append(f'  {rank:<5} {str(lbl_c[i]):<28} {dmg[i]:>8.4f}  '
                         f'{deg_arr[i]:>8.4f} {btw_arr[i]:>12.4f} '
                         f'{cls_arr[i]:>10.4f} {eig_arr[i]:>12.4f}')
        lines.append('')

    report = '\n'.join(lines)
    print('\n' + report)

    txt_out = os.path.join(FIGS_DIR, 'centrality_vulnerability_correlation.txt')
    with open(txt_out, 'w') as f:
        f.write(report)
    print(f'\nSaved: {txt_out}')

    print('\nDone.')
