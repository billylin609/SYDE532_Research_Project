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
# # Alzheimer's Disease — Two Network Animations

# %%
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['VTK_DEFAULT_RENDER_WINDOW_TYPE'] = 'OSMesa'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import numpy as np

from IPython.display import Image, display

from HCP_Animations.model import (
    load_sc_layout, tau_trajectory, clinical_label, run_motter_lai_trace,
    build_ml_frame_info, TAU_CMAP, ML_CMAP,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

np.random.seed(42)
print('Imports OK')

# %% — Setup: load data
sc_ctx, labels, N, pos, braak_stage, node_size, W_MAX, EDGES = load_sc_layout()
print('Braak stage counts:', dict(zip(*np.unique(braak_stage, return_counts=True))))
print('Layout ready for', N, 'nodes')

TOTAL_FRAMES = 45
stage_bar_colors = {0: '#4e8ec9', 1: '#fd8d3c', 2: '#d73027', 3: '#67000d'}

KEY_LABELS = {'L_entorhinal': 'Entorhinal\n(Braak I)',
              'L_parahippocampal': 'Parahippo.',
              'L_isthmuscingulate': 'Isthmus\nCing.',
              'L_posteriorcingulate': 'Post.\nCing.',
              'L_precuneus': 'Precuneus',
              'L_inferiorparietal': 'Inf.\nParietal'}

# %% — Animation 1: Biological AD Tau Spreading
print('Generating Animation 1: Biological AD Tau Spreading...')

fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor('#111111')
ax.set_facecolor('#111111')
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.12, 1.15)
ax.set_aspect('equal')
ax.axis('off')

fig.suptitle('Alzheimer\'s Disease — Biological Tau Spreading (Braak Staging)',
             color='white', fontsize=13, fontweight='bold', y=0.97)

for (i, j) in EDGES:
    xi, yi = pos[i]; xj, yj = pos[j]
    w = sc_ctx[i, j]
    alp = 0.04 + (0.18 - 0.04) * (w / W_MAX)
    ax.plot([xi, xj], [yi, yj], '-', color='#aaaaaa', alpha=alp, lw=0.4, zorder=2)

for sign in [-1, 1]:
    e = Ellipse((sign * 0.48, 0.05), width=1.10, height=2.10,
                linewidth=1.2, edgecolor='#444444', facecolor='none', zorder=1)
    ax.add_patch(e)
ax.axvline(0, color='#333333', lw=1.0, ls='--', zorder=1)
ax.text(0, 1.08, 'Anterior', ha='center', va='bottom', fontsize=7, color='#777777')
ax.text(0, -1.06, 'Posterior', ha='center', va='top', fontsize=7, color='#777777')
ax.text(-1.08, 0, 'L', ha='right', va='center', fontsize=9, color='#777777', fontweight='bold')
ax.text(1.08, 0, 'R', ha='left', va='center', fontsize=9, color='#777777', fontweight='bold')

node_x = np.array([pos[i][0] for i in range(N)])
node_y = np.array([pos[i][1] for i in range(N)])

scatter = ax.scatter(node_x, node_y, s=node_size,
                     c=np.zeros(N), cmap=TAU_CMAP, vmin=0, vmax=1,
                     edgecolors='white', linewidths=0.5, zorder=5)

for lab_key, txt in KEY_LABELS.items():
    if lab_key in labels:
        idx = labels.index(lab_key)
        ax.text(pos[idx][0] - 0.04, pos[idx][1] + 0.06, txt,
                fontsize=6, color='#dddddd', ha='center', zorder=6,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

title_txt = ax.text(0, -1.00, '', ha='center', va='top',
                    fontsize=9, color='white', fontweight='bold',
                    bbox=dict(facecolor='#222222', edgecolor='none', pad=4))
frame_txt = ax.text(-1.10, 1.05, '', ha='left', va='top', fontsize=8, color='#aaaaaa')

sm = cm.ScalarMappable(cmap=TAU_CMAP, norm=Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
cbar.set_label('Tau Burden', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)

leg_patches = [
    mpatches.Patch(color=TAU_CMAP(0.0), label='Healthy'),
    mpatches.Patch(color=TAU_CMAP(0.35), label='Braak I-II (Entorhinal)'),
    mpatches.Patch(color=TAU_CMAP(0.65), label='Braak III-IV (Limbic)'),
    mpatches.Patch(color=TAU_CMAP(1.00), label='Braak V-VI (Isocortical)'),
]
ax.legend(handles=leg_patches, loc='upper left', fontsize=7,
          facecolor='#222222', edgecolor='#555555', labelcolor='white')


def update_bio(frame):
    tau = np.array([tau_trajectory(braak_stage[i], frame, TOTAL_FRAMES) for i in range(N)])
    scatter.set_array(tau)
    sizes = node_size.copy()
    if 5 <= frame < 15:
        active = braak_stage == 1
    elif 15 <= frame < 25:
        active = braak_stage == 2
    elif 25 <= frame < 39:
        active = braak_stage == 3
    else:
        active = np.zeros(N, dtype=bool)
    pulse = 1.0 + 0.4 * np.sin(frame * 0.7)
    sizes[active] *= pulse
    scatter.set_sizes(sizes)
    title_txt.set_text(clinical_label(frame))
    frame_txt.set_text(f'Frame {frame+1}/{TOTAL_FRAMES}')
    return scatter, title_txt, frame_txt


anim1 = animation.FuncAnimation(fig, update_bio, frames=TOTAL_FRAMES, interval=180, blit=True)
writer = animation.PillowWriter(fps=8)
anim1.save(os.path.join(FIGS_DIR, 'ad_bio_animation.gif'), writer=writer, dpi=130)
plt.close(fig)
print('Saved: ad_bio_animation.gif')

# %%
display(Image(os.path.join(FIGS_DIR, 'ad_bio_animation.gif')))

# %% — Animation 2: Motter-Lai Cascade from Entorhinal Cortex
ALPHA_ANIM = 0.3
entorhinal_seed = labels.index('L_entorhinal')

print(f'Running Motter-Lai cascade from L_entorhinal (idx {entorhinal_seed}), α={ALPHA_ANIM}...')
ml_history, ml_loads, ml_C = run_motter_lai_trace(sc_ctx, entorhinal_seed, ALPHA_ANIM)

n_rounds = len(ml_history)
all_failed = set()
for s in ml_history:
    all_failed.update(s)

print(f'Cascade rounds: {n_rounds}')
print(f'Total nodes failed: {len(all_failed)} / {N}')

frames_info, ROUND_FRAMES = build_ml_frame_info(n_rounds)
TOTAL_FRAMES2 = len(frames_info)
print(f'Animation 2 total frames: {TOTAL_FRAMES2}')


def frame_state(fi):
    """Return (failed, load_dict, collapsing, txt) for a given frame index."""
    info = frames_info[fi]
    phase, rnd, sub = info['phase'], info['round'], info['sub']

    if phase == 'A':
        failed = set()
        load_dict = ml_loads[0]
        txt = f'Baseline healthy network  |  α = {ALPHA_ANIM}'
    elif phase == 'B':
        failed = {entorhinal_seed}
        load_dict = ml_loads[0]
        txt = 'Entorhinal cortex fails (AD epicentre)  |  Load redistributes...'
    elif phase == 'C':
        failed = set()
        for r2 in range(rnd):
            failed.update(ml_history[r2])
        load_dict = ml_loads[min(rnd - 1, len(ml_loads) - 1)]
        collapsing = ml_history[rnd] if rnd < len(ml_history) else set()
        txt = (f'Cascade round {rnd}  |  {len(collapsing)} node(s) overloaded  |  '
               f'Total failed: {len(failed) + (len(collapsing) if sub >= ROUND_FRAMES-1 else 0)}')
        if sub == ROUND_FRAMES - 1:
            failed.update(collapsing)
        return failed, load_dict, collapsing if sub < ROUND_FRAMES - 1 else set(), txt
    else:
        failed = all_failed.copy()
        load_dict = ml_loads[-1] if ml_loads else {}
        surviving = N - len(failed)
        txt = f'Cascade stable  |  {surviving}/{N} regions survive ({surviving/N*100:.0f}%)'

    return failed, load_dict, set(), txt


print('Generating Animation 2: Motter-Lai Cascade...')

fig2, ax2 = plt.subplots(figsize=(11, 8))
fig2.patch.set_facecolor('#0d0d0d')
ax2.set_facecolor('#0d0d0d')
ax2.set_xlim(-1.15, 1.15)
ax2.set_ylim(-1.12, 1.15)
ax2.set_aspect('equal')
ax2.axis('off')

fig2.suptitle(
    f'Motter–Lai Cascade from Entorhinal Cortex (AD Epicentre) — α = {ALPHA_ANIM}',
    color='white', fontsize=12, fontweight='bold', y=0.97)

for sign in [-1, 1]:
    e = Ellipse((sign * 0.48, 0.05), width=1.10, height=2.10,
                linewidth=1.2, edgecolor='#2a2a2a', facecolor='none', zorder=1)
    ax2.add_patch(e)
ax2.axvline(0, color='#1a1a1a', lw=1.0, ls='--', zorder=1)
ax2.text(0, 1.08, 'Anterior', ha='center', va='bottom', fontsize=7, color='#555555')
ax2.text(0, -1.06, 'Posterior', ha='center', va='top', fontsize=7, color='#555555')
ax2.text(-1.10, 0, 'L', ha='right', va='center', fontsize=9, color='#555555', fontweight='bold')
ax2.text(1.10, 0, 'R', ha='left', va='center', fontsize=9, color='#555555', fontweight='bold')

for (i, j) in EDGES:
    xi, yi = pos[i]; xj, yj = pos[j]
    w = sc_ctx[i, j]
    alp = 0.02 + (0.12 - 0.02) * (w / W_MAX)
    ax2.plot([xi, xj], [yi, yj], '-', color='#555566', alpha=alp, lw=0.35, zorder=2)

node_x2 = np.array([pos[i][0] for i in range(N)])
node_y2 = np.array([pos[i][1] for i in range(N)])

scatter2 = ax2.scatter(node_x2, node_y2, s=node_size,
                        c=np.zeros(N), cmap=ML_CMAP, vmin=0, vmax=1.5,
                        edgecolors='#aaaaaa', linewidths=0.5, zorder=5)

LABEL_NODES = {'L_entorhinal': 'Entorhinal\n(Seed)', 'L_precuneus': 'Precuneus',
               'L_posteriorcingulate': 'Post.Cing.', 'L_inferiorparietal': 'Inf.Par.'}
for lab_key, txt in LABEL_NODES.items():
    if lab_key in labels:
        idx = labels.index(lab_key)
        ax2.text(pos[idx][0] - 0.04, pos[idx][1] + 0.07, txt,
                 fontsize=6.5, color='#eeeeee', ha='center', zorder=7,
                 path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

status_txt = ax2.text(0, -1.00, '', ha='center', va='top', fontsize=9,
                       color='white', fontweight='bold',
                       bbox=dict(facecolor='#1a1a1a', edgecolor='none', pad=4))
frame_txt2 = ax2.text(-1.10, 1.06, '', ha='left', va='top', fontsize=8, color='#666666')

failed_scatter = ax2.scatter([], [], s=node_size.mean(), marker='x',
                              color='#ff3333', linewidths=1.5, zorder=8)
warn_scatter = ax2.scatter([], [], s=node_size.mean() * 2.5, marker='o',
                            facecolors='none', edgecolors='#ffff00',
                            linewidths=2.0, zorder=7, alpha=0.9)

sm2 = cm.ScalarMappable(cmap=ML_CMAP, norm=Normalize(0, 1.5))
sm2.set_array([])
cbar2 = fig2.colorbar(sm2, ax=ax2, fraction=0.02, pad=0.01)
cbar2.set_label('Load fraction  L/C', color='white', fontsize=9)
cbar2.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white', fontsize=8)
cbar2.ax.axhline(1.0, color='yellow', lw=1.5, ls='--')
cbar2.ax.text(2.1, 1.0, 'Capacity\nlimit', color='yellow', fontsize=7, va='center')

leg2_patches = [
    mpatches.Patch(color=ML_CMAP(0.0), label='Low load (healthy)'),
    mpatches.Patch(color=ML_CMAP(0.55), label='Moderate load'),
    mpatches.Patch(color=ML_CMAP(0.85), label='Near capacity'),
    mpatches.Patch(color=ML_CMAP(1.0), label='Overloaded (failing)'),
    mpatches.Patch(color='#666666', label='Failed / removed'),
    mpatches.Patch(facecolor='none', edgecolor='yellow', label='Warning: about to fail'),
]
ax2.legend(handles=leg2_patches, loc='upper left', fontsize=7,
           facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')


def update_ml(fi):
    info = frames_info[fi]
    failed, load_dict, collapsing, txt = frame_state(fi)

    c_vals = np.zeros(N)
    sz = node_size.copy()
    edgecol = ['#aaaaaa'] * N

    for i in range(N):
        if i in failed:
            c_vals[i] = 0.0
            sz[i] = node_size[i] * 0.3
            edgecol[i] = '#333333'
        elif i in collapsing:
            pulse = 0.8 + 0.4 * np.sin(fi * 1.8)
            c_vals[i] = min(1.5, 1.2 * pulse)
            sz[i] = node_size[i] * (1.5 + 0.5 * np.sin(fi * 1.8))
            edgecol[i] = '#ffff00'
        else:
            fl = load_dict.get(i, 0.0)
            c_vals[i] = min(1.5, fl)

    scatter2.set_array(c_vals)
    scatter2.set_sizes(sz)
    scatter2.set_edgecolors(edgecol)

    failed_list = list(failed)
    if failed_list:
        fx = [pos[i][0] for i in failed_list]
        fy = [pos[i][1] for i in failed_list]
        failed_scatter.set_offsets(np.c_[fx, fy])
        failed_scatter.set_sizes([node_size[i] * 0.4 for i in failed_list])
    else:
        failed_scatter.set_offsets(np.empty((0, 2)))

    warn_list = list(collapsing)
    if warn_list:
        wx = [pos[i][0] for i in warn_list]
        wy = [pos[i][1] for i in warn_list]
        warn_scatter.set_offsets(np.c_[wx, wy])
        warn_scatter.set_sizes([node_size[i] * 2.0 for i in warn_list])
    else:
        warn_scatter.set_offsets(np.empty((0, 2)))

    status_txt.set_text(txt)
    frame_txt2.set_text(f'Frame {fi+1}/{TOTAL_FRAMES2}')
    return scatter2, failed_scatter, warn_scatter, status_txt, frame_txt2


anim2 = animation.FuncAnimation(fig2, update_ml, frames=TOTAL_FRAMES2, interval=220, blit=True)
writer2 = animation.PillowWriter(fps=5)
anim2.save(os.path.join(FIGS_DIR, 'ad_motterlai_animation.gif'), writer=writer2, dpi=130)
plt.close(fig2)
print('Saved: ad_motterlai_animation.gif')

# %%
display(Image(os.path.join(FIGS_DIR, 'ad_motterlai_animation.gif')))

# %% — Side-by-Side Static Comparison (Final Frame)
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
fig3.patch.set_facecolor('#111111')

titles = ['Biological AD Progression (Braak VI — End-stage)',
          f'Motter–Lai Cascade (α={ALPHA_ANIM}) — Post-cascade Stable State']
cmaps = [TAU_CMAP, ML_CMAP]
vmaxs = [1.0, 1.5]
clabels = ['Tau Burden (0=healthy, 1=fully affected)',
           'Load fraction L/C (>1 = overloaded → removed)']

for ax_idx, ax3 in enumerate(axes3):
    ax3.set_facecolor('#111111')
    ax3.set_xlim(-1.15, 1.15)
    ax3.set_ylim(-1.12, 1.15)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title(titles[ax_idx], color='white', fontsize=10, pad=10)

    for sign in [-1, 1]:
        e = Ellipse((sign * 0.48, 0.05), width=1.10, height=2.10,
                    linewidth=1.2, edgecolor='#333333', facecolor='none', zorder=1)
        ax3.add_patch(e)
    ax3.axvline(0, color='#222222', lw=1.0, ls='--', zorder=1)

    for (i, j) in EDGES:
        xi, yi = pos[i]; xj, yj = pos[j]
        w = sc_ctx[i, j]
        alp = 0.03 + (0.15 - 0.03) * (w / W_MAX)
        ax3.plot([xi, xj], [yi, yj], '-', color='#666666', alpha=alp, lw=0.3, zorder=2)

    if ax_idx == 0:
        tau_final = np.array([tau_trajectory(braak_stage[i], TOTAL_FRAMES - 1, TOTAL_FRAMES) for i in range(N)])
        sc3 = ax3.scatter(node_x, node_y, s=node_size, c=tau_final,
                          cmap=TAU_CMAP, vmin=0, vmax=1.0,
                          edgecolors='white', linewidths=0.4, zorder=5)
        cb3 = fig3.colorbar(sc3, ax=ax3, fraction=0.025, pad=0.01)
        cb3.set_label(clabels[0], color='white', fontsize=8)
        plt.setp(cb3.ax.yaxis.get_ticklabels(), color='white', fontsize=7)
        cb3.ax.yaxis.set_tick_params(color='white')

        stage_labels_map = {1: 'I-II', 2: 'III-IV', 3: 'V-VI'}
        stage_cols_map = {1: '#fd8d3c', 2: '#d73027', 3: '#67000d'}
        for st_val, st_lab in stage_labels_map.items():
            st_nodes = [i for i in range(N) if braak_stage[i] == st_val]
            if st_nodes:
                cx = np.mean([node_x[i] for i in st_nodes])
                cy = np.mean([node_y[i] for i in st_nodes])
                ax3.text(cx, cy, f'Braak {st_lab}', fontsize=7, color='white',
                         ha='center', va='center', fontweight='bold',
                         bbox=dict(facecolor=stage_cols_map[st_val], alpha=0.7, pad=2), zorder=8)
    else:
        c_final = np.zeros(N)
        final_load = ml_loads[-1] if ml_loads else {}
        sz_final = node_size.copy()
        edge_fin = ['#aaaaaa'] * N
        for i in range(N):
            if i in all_failed:
                c_final[i] = 0.0
                sz_final[i] = node_size[i] * 0.25
                edge_fin[i] = '#333333'
            else:
                c_final[i] = min(1.5, final_load.get(i, 0.0))

        sc3b = ax3.scatter(node_x, node_y, s=sz_final, c=c_final,
                           cmap=ML_CMAP, vmin=0, vmax=1.5,
                           edgecolors=edge_fin, linewidths=0.5, zorder=5)
        fail_xy = np.array([[pos[i][0], pos[i][1]] for i in all_failed])
        if len(fail_xy):
            ax3.scatter(fail_xy[:, 0], fail_xy[:, 1], s=node_size.mean() * 0.5,
                        marker='x', color='#ff4444', linewidths=1.5, zorder=8)

        ax3.scatter([pos[entorhinal_seed][0]], [pos[entorhinal_seed][1]],
                    s=200, marker='*', color='#ffdd00', zorder=9)
        ax3.text(pos[entorhinal_seed][0], pos[entorhinal_seed][1] + 0.09,
                 'Seed\n(Entorhinal)', color='#ffdd00', fontsize=7, ha='center',
                 path_effects=[pe.withStroke(linewidth=1.5, foreground='black')], zorder=10)

        cb3b = fig3.colorbar(sc3b, ax=ax3, fraction=0.025, pad=0.01)
        cb3b.set_label(clabels[1], color='white', fontsize=8)
        plt.setp(cb3b.ax.yaxis.get_ticklabels(), color='white', fontsize=7)
        cb3b.ax.yaxis.set_tick_params(color='white')
        cb3b.ax.axhline(1.0, color='yellow', lw=1.2, ls='--')

        surviving = N - len(all_failed)
        ax3.text(0, -1.05, f'{surviving}/{N} regions survive ({surviving/N*100:.0f}%)',
                 ha='center', va='top', color='#aaffaa', fontsize=9, fontweight='bold')

# Use node_x/node_y from above (static)
node_x = np.array([pos[i][0] for i in range(N)])
node_y = np.array([pos[i][1] for i in range(N)])

plt.suptitle('Final State Comparison — AD Biology (left) vs Motter–Lai Cascade (right)',
             color='white', fontsize=12, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'ad_final_comparison.png'), dpi=150, bbox_inches='tight',
            facecolor='#111111')
plt.show()
print('Saved: ad_final_comparison.png')

# %%
display(Image(os.path.join(FIGS_DIR, 'ad_final_comparison.png')))
