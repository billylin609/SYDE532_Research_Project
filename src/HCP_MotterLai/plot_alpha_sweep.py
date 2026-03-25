"""Bar plot of G vs alpha for hub attack — shows tipping point."""
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from HCP_MotterLai.model import load_sc_and_loads, motter_lai

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

BG    = '#0a0a12'
PANEL = '#0f0f1a'

# ── Data ──────────────────────────────────────────────────────────────────────
sc_ctx, sc_ctx_labels, N, A_bin, G_w, G_uw, L0_uw_arr, L0_w_arr, degree_arr = load_sc_and_loads()
attack = int(np.argmax(L0_uw_arr))
attack_name = str(sc_ctx_labels[attack])

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
          0.55, 0.60, 0.70, 0.80, 1.00]

print(f'Sweeping α for hub attack: {attack_name}')
G_vals, failed_vals, round_vals = [], [], []
for a in ALPHAS:
    G, nf, h = motter_lai(sc_ctx, attack, a, return_history=True)
    G_vals.append(G)
    failed_vals.append(nf)
    round_vals.append(len(h) - 1)
    print(f'  α={a:.2f}  G={G:.4f}  failed={nf:2d}  rounds={len(h)-1}')

G_vals = np.array(G_vals)

# ── Bar colours by survival ────────────────────────────────────────────────────
def bar_color(g):
    if g < 0.50:  return '#e74c3c'   # catastrophic
    if g < 0.85:  return '#f39c12'   # partial cascade
    if g < 0.97:  return '#27ae60'   # contained
    return '#4A90D9'                  # negligible (only seed)

colors = [bar_color(g) for g in G_vals]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
ax.set_facecolor(PANEL)

x = np.arange(len(ALPHAS))
bars = ax.bar(x, G_vals, color=colors, edgecolor='#223344', linewidth=0.8,
              width=0.65, zorder=3)

# Tipping-point region shading
tip_lo = ALPHAS.index(0.20)
tip_hi = ALPHAS.index(0.25)
ax.axvspan(tip_lo + 0.325, tip_hi - 0.325 + 0.65, color='white', alpha=0.06,
           label='Tipping zone', zorder=2)
ax.axvline(tip_lo + 0.325, color='white', lw=1.2, ls='--', alpha=0.5, zorder=4)
ax.axvline(tip_hi + 0.325, color='white', lw=1.2, ls='--', alpha=0.5, zorder=4)
ax.text((tip_lo + tip_hi) / 2 + 0.65, 0.55, 'Tipping\npoint',
        ha='center', va='center', color='white', fontsize=8, alpha=0.8,
        path_effects=[pe.withStroke(linewidth=1.5, foreground=PANEL)])

# Annotate our two chosen α values
for chosen_a, label, yoff in [(0.10, 'Bad  α=0.10\n(catastrophic)', -0.18),
                                (0.40, 'Good  α=0.40\n(contained)', 0.08)]:
    xi = ALPHAS.index(chosen_a)
    g  = G_vals[xi]
    col = bar_color(g)
    ax.annotate(label,
                xy=(xi, g), xytext=(xi, g + yoff + 0.06),
                ha='center', va='bottom', fontsize=8.5, color=col, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=1.5, foreground=PANEL)],
                arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

# Failed-count labels inside bars
for i, (g, nf) in enumerate(zip(G_vals, failed_vals)):
    if nf > 1:
        ax.text(i, max(0.03, g - 0.06), f'{nf} fail',
                ha='center', va='bottom', fontsize=7, color='white',
                path_effects=[pe.withStroke(linewidth=1, foreground='black')])

# Threshold lines
ax.axhline(0.50, color='#e74c3c', lw=1.0, ls=':', alpha=0.6)
ax.axhline(0.85, color='#f39c12', lw=1.0, ls=':', alpha=0.6)
ax.axhline(0.97, color='#4A90D9', lw=1.0, ls=':', alpha=0.6)

# Legend patches
from matplotlib.patches import Patch
handles = [
    Patch(facecolor='#e74c3c', label='Catastrophic  (G < 0.50)'),
    Patch(facecolor='#f39c12', label='Partial cascade  (0.50 ≤ G < 0.85)'),
    Patch(facecolor='#27ae60', label='Contained  (0.85 ≤ G < 0.97)'),
    Patch(facecolor='#4A90D9', label='Negligible  (G ≥ 0.97)'),
]
ax.legend(handles=handles, loc='upper left', fontsize=8,
          facecolor='#1a1a2a', edgecolor='#334455',
          labelcolor='white', framealpha=0.95)

ax.set_xticks(x)
ax.set_xticklabels([f'{a:.2f}' for a in ALPHAS], color='#aabbcc', fontsize=9)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_yticklabels([f'{v:.1f}' for v in np.arange(0, 1.1, 0.1)], color='#aabbcc', fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_xlabel('Tolerance parameter  α', color='#aabbcc', fontsize=11)
ax.set_ylabel('Network survival  G = N\'/N', color='#aabbcc', fontsize=11)
ax.set_title(f'Motter–Lai: Network Survival vs Tolerance  |  Hub attack: {attack_name}  (L₀ = {L0_uw_arr[attack]:.0f})',
             color='white', fontsize=12, fontweight='bold', pad=10)
ax.grid(axis='y', color='#334455', lw=0.6, alpha=0.6, zorder=1)
for spine in ax.spines.values():
    spine.set_edgecolor('#334455')

plt.tight_layout()
out = os.path.join(FIGS_DIR, 'hcp_ml_alpha_sweep.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f'\nSaved: {out}')
