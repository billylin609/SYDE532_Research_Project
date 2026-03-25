"""Cascade damage vs tolerance — L vs R superiorparietal comparison."""
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from HCP_MotterLai.model import load_sc_and_loads, motter_lai

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

sc_ctx, sc_ctx_labels, N, A_bin, G_w, G_uw, L0_uw_arr, L0_w_arr, degree_arr = load_sc_and_loads()

node_R = int(np.argmax(L0_uw_arr))
node_L = next(i for i, l in enumerate(sc_ctx_labels) if 'L_superiorparietal' in str(l))

ALPHAS = np.linspace(0.0, 1.0, 40)

print('Sweeping α for R_superiorparietal…')
G_R = [motter_lai(sc_ctx, node_R, a)[0] for a in ALPHAS]

print('Sweeping α for L_superiorparietal…')
G_L = [motter_lai(sc_ctx, node_L, a)[0] for a in ALPHAS]

dmg_R = 1 - np.array(G_R)
dmg_L = 1 - np.array(G_L)

# Critical alpha = first alpha where damage < 0.05
def crit_alpha(dmg, alphas):
    for a, d in zip(alphas, dmg):
        if d < 0.05:
            return a
    return alphas[-1]

crit_R = crit_alpha(dmg_R, ALPHAS)
crit_L = crit_alpha(dmg_L, ALPHAS)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(ALPHAS, dmg_R, 'o-', color='#e74c3c', linewidth=2, markersize=4,
        label=f'R\_superiorparietal  (L₀={L0_uw_arr[node_R]:.0f})')
ax.plot(ALPHAS, dmg_L, 's-', color='#f39c12', linewidth=2, markersize=4,
        label=f'L\_superiorparietal  (L₀={L0_uw_arr[node_L]:.0f})')

ax.axvline(crit_R, color='#e74c3c', lw=1.5, ls=':', alpha=0.8,
           label=f'Critical α (R) ≈ {crit_R:.2f}')
ax.axvline(crit_L, color='#f39c12', lw=1.5, ls=':', alpha=0.8,
           label=f'Critical α (L) ≈ {crit_L:.2f}')

# Mark chosen scenarios
for node, alpha, g, color, label in [
    (node_R, 0.10, dmg_R[np.argmin(np.abs(ALPHAS-0.10))], '#e74c3c', 'R  α=0.10'),
    (node_R, 0.40, dmg_R[np.argmin(np.abs(ALPHAS-0.40))], '#e74c3c', 'R  α=0.40'),
    (node_L, 0.10, dmg_L[np.argmin(np.abs(ALPHAS-0.10))], '#f39c12', 'L  α=0.10'),
    (node_L, 0.40, dmg_L[np.argmin(np.abs(ALPHAS-0.40))], '#f39c12', 'L  α=0.40'),
]:
    ax.scatter([alpha], [g], s=100, color=color, zorder=5, edgecolors='black', lw=0.8)
    va = 'bottom' if g < 0.85 else 'top'
    ax.annotate(label, (alpha, g), xytext=(alpha + 0.02, g + 0.03),
                fontsize=8, color=color, fontweight='bold')

ax.axhline(0.0, color='black', lw=0.8, ls='--', alpha=0.4)
ax.set_xlabel('Tolerance parameter  α', fontsize=12)
ax.set_ylabel('Cascade damage  =  1 − G', fontsize=12)
ax.set_title('Cascade Damage vs Tolerance\nHub Attack: L vs R Superiorparietal',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(FIGS_DIR, 'hcp_ml_alpha_sweep_compare.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
