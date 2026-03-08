from enigmatoolbox.datasets import load_sc, load_fc
from enigmatoolbox.utils.parcellation import parcel_to_surface
from nilearn import plotting, datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.cm as mplcm
import numpy as np
import os

os.makedirs('./figs', exist_ok=True)
os.makedirs('./figs/_tmp', exist_ok=True)

sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()
fc_ctx, fc_ctx_labels, fc_sctx, fc_sctx_labels = load_fc()

# Fetch fsaverage5 surface meshes from nilearn
fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

# ── Render 4 views to temp PNGs via nilearn ──────────────────────────────────
def render_views(lh, rh, cmap, vmin, vmax, prefix):
    """Save lateral-L and lateral-R PNGs; return their paths."""
    config = [
        ('pial_left',  'sulc_left',  lh, 'left',  'lateral'),
        ('pial_right', 'sulc_right', rh, 'right', 'lateral'),
    ]
    paths = []
    for i, (mesh_key, sulc_key, data, hemi, view) in enumerate(config):
        path = f'./figs/_tmp/{prefix}_{i}.png'
        plotting.plot_surf_stat_map(
            surf_mesh=fsaverage[mesh_key],
            stat_map=data,
            hemi=hemi,
            view=view,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            colorbar=False,
            bg_map=fsaverage[sulc_key],
            bg_on_data=True,
            output_file=path,
        )
        plt.close('all')
        paths.append(path)
        print(f"  view {i+1}/4 done")
    return paths

# ── Seed 1: L_middletemporal (cortical → cortical) ───────────────────────────
print("\nSeed 1: L_middletemporal")
seed_ctx = "L_middletemporal"
seed_conn_fc = fc_ctx[[i for i, item in enumerate(fc_ctx_labels) if seed_ctx in item], ].squeeze()
seed_conn_sc = sc_ctx[[i for i, item in enumerate(sc_ctx_labels) if seed_ctx in item], ].squeeze()

seed_conn_fc_fsa5 = parcel_to_surface(seed_conn_fc, 'aparc_fsa5')
seed_conn_sc_fsa5 = parcel_to_surface(seed_conn_sc, 'aparc_fsa5')

n_v = len(seed_conn_fc_fsa5) // 2
lh_fc1, rh_fc1 = seed_conn_fc_fsa5[:n_v], seed_conn_fc_fsa5[n_v:]
lh_sc1, rh_sc1 = seed_conn_sc_fsa5[:n_v], seed_conn_sc_fsa5[n_v:]

print("  Rendering FC views...")
paths_fc1 = render_views(lh_fc1, rh_fc1, 'Reds',  0.2, 0.7,  'ctx_fc')
print("  Rendering SC views...")
paths_sc1 = render_views(lh_sc1, rh_sc1, 'Blues', 2,   10,   'ctx_sc')

# ── Seed 2: Lhippo (subcortical → cortical) ──────────────────────────────────
print("\nSeed 2: Lhippo")
seed_sctx = "Lhippo"
seed_conn_fc = fc_sctx[[i for i, item in enumerate(fc_sctx_labels) if seed_sctx in item], ].squeeze()
seed_conn_sc = sc_sctx[[i for i, item in enumerate(sc_sctx_labels) if seed_sctx in item], ].squeeze()

seed_conn_fc_fsa5 = parcel_to_surface(seed_conn_fc, 'aparc_fsa5')
seed_conn_sc_fsa5 = parcel_to_surface(seed_conn_sc, 'aparc_fsa5')

n_v = len(seed_conn_fc_fsa5) // 2
lh_fc2, rh_fc2 = seed_conn_fc_fsa5[:n_v], seed_conn_fc_fsa5[n_v:]
lh_sc2, rh_sc2 = seed_conn_sc_fsa5[:n_v], seed_conn_sc_fsa5[n_v:]

print("  Rendering FC views...")
paths_fc2 = render_views(lh_fc2, rh_fc2, 'Reds',  0.1, 0.3,  'sctx_fc')
print("  Rendering SC views...")
paths_sc2 = render_views(lh_sc2, rh_sc2, 'Blues', 1,   10,   'sctx_sc')

# ── Assemble 4-row publication figure ────────────────────────────────────────
#
#  Row 0 │ Functional  │ seed: L_middletemporal   Reds  0.2–0.7
#  Row 1 │ Structural  │ seed: L_middletemporal   Blues 2–10
#        │ ─────────── divider ─────────────────────────────── │
#  Row 2 │ Functional  │ seed: Lhippo             Reds  0.1–0.3
#  Row 3 │ Structural  │ seed: Lhippo             Blues 1–10
#
row_configs = [
    (0, paths_fc1, 'Reds',  0.2, 0.7,
     'Seed-Based Functional Cortico-Cortical Connectivity',
     'L_middletemporal', 'cortical'),
    (1, paths_sc1, 'Blues', 2,   10,
     'Seed-Based Structural Cortico-Cortical Connectivity',
     'L_middletemporal', 'cortical'),
    (2, paths_fc2, 'Reds',  0.1, 0.3,
     'Seed-Based Functional Subcortico-Cortical Connectivity',
     'Left hippocampus (Lhippo)', 'subcortical'),
    (3, paths_sc2, 'Blues', 1,   10,
     'Seed-Based Structural Subcortico-Cortical Connectivity',
     'Left hippocampus (Lhippo)', 'subcortical'),
]

fig = plt.figure(figsize=(22, 12), facecolor='white')

# 2 rows (cortical / subcortical seed) × (FC left | FC right | SC left | SC right | colorbar pair)
# Laid out as: [FC-LH | FC-RH | gap | SC-LH | SC-RH | cb-FC | cb-SC]
gs = GridSpec(2, 7, figure=fig,
              width_ratios=[5, 5, 0.3, 5, 5, 0.35, 0.35],
              height_ratios=[1, 1],
              hspace=0.30, wspace=0.05,
              left=0.02, right=0.98, top=0.93, bottom=0.04)

# Row 0 = cortical seed (L_middletemporal), Row 1 = subcortical seed (Lhippo)
# Within each row: FC-LH | FC-RH | [gap] | SC-LH | SC-RH | cb-FC | cb-SC
seed_rows = [
    (0, paths_fc1, paths_sc1, 0.2, 0.7, 2, 10,
     'L_middletemporal', 'cortical',
     'Functional Cortico-Cortical Connectivity',
     'Structural Cortico-Cortical Connectivity'),
    (1, paths_fc2, paths_sc2, 0.1, 0.3, 1, 10,
     'Left Hippocampus (Lhippo)', 'subcortical',
     'Functional Subcortico-Cortical Connectivity',
     'Structural Subcortico-Cortical Connectivity'),
]

for row, p_fc, p_sc, fc_min, fc_max, sc_min, sc_max, seed_name, seed_type, fc_title, sc_title in seed_rows:
    imgs_fc = [mpimg.imread(p) for p in p_fc]   # [LH, RH]
    imgs_sc = [mpimg.imread(p) for p in p_sc]   # [LH, RH]

    # columns: 0=FC-LH, 1=FC-RH, 2=gap, 3=SC-LH, 4=SC-RH, 5=cb-FC, 6=cb-SC
    panels = [
        (0, imgs_fc[0], 'Left Hemisphere',  fc_title, True),
        (1, imgs_fc[1], 'Right Hemisphere', '',        False),
        (3, imgs_sc[0], 'Left Hemisphere',  sc_title,  True),
        (4, imgs_sc[1], 'Right Hemisphere', '',        False),
    ]

    for col, img, hemi_label, subtitle, show_seed in panels:
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.axis('off')

        ax.text(0.5, -0.03, hemi_label, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, color='#555', style='italic')

        if subtitle:
            ax.set_title(
                f'{subtitle}\n' + r'$\it{Seed}$' + f': {seed_name}',
                fontsize=10, fontweight='bold', loc='left', pad=5, color='#111',
            )

        if show_seed:
            h, w = img.shape[:2]
            sx = int(w * 0.30) if seed_type == 'cortical' else int(w * 0.38)
            sy = int(h * 0.58) if seed_type == 'cortical' else int(h * 0.63)
            ax.plot(sx, sy, 'o', mfc='none', mec='black', ms=8, mew=1.5, zorder=5)
            ax.annotate('seed', xy=(sx, sy),
                        xytext=(sx - int(w * 0.13), sy + int(h * 0.15)),
                        fontsize=9, color='black',
                        arrowprops=dict(arrowstyle='-', color='black', lw=1.0))

    # FC colorbar
    ax_cb_fc = fig.add_subplot(gs[row, 5])
    sm_fc = mplcm.ScalarMappable(cmap='Reds', norm=Normalize(vmin=fc_min, vmax=fc_max))
    sm_fc.set_array([])
    cb_fc = plt.colorbar(sm_fc, cax=ax_cb_fc)
    cb_fc.set_ticks([fc_min, fc_max])
    cb_fc.ax.tick_params(labelsize=8)

    # SC colorbar
    ax_cb_sc = fig.add_subplot(gs[row, 6])
    sm_sc = mplcm.ScalarMappable(cmap='Blues', norm=Normalize(vmin=sc_min, vmax=sc_max))
    sm_sc.set_array([])
    cb_sc = plt.colorbar(sm_sc, cax=ax_cb_sc)
    cb_sc.set_ticks([sc_min, sc_max])
    cb_sc.ax.tick_params(labelsize=8)

# Vertical divider between FC and SC panels
fig.add_artist(plt.Line2D(
    [0.485, 0.485], [0.04, 0.96],
    transform=fig.transFigure,
    color='#bbb', linewidth=1.2, linestyle='--'
))

# Horizontal divider between cortical and subcortical seed rows
fig.add_artist(plt.Line2D(
    [0.02, 0.98], [0.50, 0.50],
    transform=fig.transFigure,
    color='#bbb', linewidth=1.2, linestyle='--'
))

plt.savefig('./figs/enigma_seed_surface.png', dpi=180,
            bbox_inches='tight', facecolor='white')
print("\n✓ Saved: ./figs/enigma_seed_surface.png")
