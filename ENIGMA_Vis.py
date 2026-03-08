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
    """Save lat-L / med-L / med-R / lat-R PNGs; return their paths."""
    config = [
        ('pial_left',  'sulc_left',  lh, 'left',  'lateral'),
        ('pial_left',  'sulc_left',  lh, 'left',  'medial'),
        ('pial_right', 'sulc_right', rh, 'right', 'medial'),
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

fig = plt.figure(figsize=(20, 18), facecolor='white')

# 4 brain rows + 1 col for colorbar; extra vertical space between the two seed pairs
gs = GridSpec(4, 5, figure=fig,
              width_ratios=[4, 4, 4, 4, 0.35],
              height_ratios=[1, 1, 1, 1],
              hspace=0.22, wspace=0.04,
              left=0.01, right=0.98, top=0.96, bottom=0.02)

for row, paths, cmap, vmin, vmax, title, seed_name, seed_type in row_configs:
    imgs = [mpimg.imread(p) for p in paths]

    for col, img in enumerate(imgs):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.axis('off')

        # Row title + seed label (left of the lateral-L view)
        if col == 0:
            ax.set_title(
                f'{title}\n'
                f'$\\it{{Seed:}}$ {seed_name}',
                fontsize=11, fontweight='bold',
                loc='left', pad=5, color='#111',
            )

        # "seed" circle + arrow on the left-lateral view
        if col == 0:
            h, w = img.shape[:2]
            # cortical seed: mid-temporal region (~30% from left, 58% from top)
            # subcortical seed: hippocampus sits slightly lower / more medial
            if seed_type == 'cortical':
                sx, sy = int(w * 0.30), int(h * 0.58)
            else:
                sx, sy = int(w * 0.38), int(h * 0.63)
            ax.plot(sx, sy, 'o', mfc='none', mec='black', ms=7, mew=1.5, zorder=5)
            ax.annotate('seed', xy=(sx, sy),
                        xytext=(sx - int(w * 0.12), sy + int(h * 0.14)),
                        fontsize=9, color='black',
                        arrowprops=dict(arrowstyle='-', color='black', lw=1.0))

    # Colorbar
    ax_cb = fig.add_subplot(gs[row, 4])
    sm = mplcm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax_cb)
    cb.ax.tick_params(labelsize=9)
    cb.set_ticks([vmin, vmax])

# Horizontal divider between the two seed pairs (between rows 1 and 2)
fig.add_artist(plt.Line2D(
    [0.01, 0.97], [0.502, 0.502],       # figure-fraction coordinates
    transform=fig.transFigure,
    color='#aaa', linewidth=1.2, linestyle='--'
))

plt.savefig('./figs/enigma_seed_surface.png', dpi=180,
            bbox_inches='tight', facecolor='white')
print("\n✓ Saved: ./figs/enigma_seed_surface.png")
