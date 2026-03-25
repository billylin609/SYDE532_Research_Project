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
# # HCP Brain Network — Motter–Lai Cascade Failure Simulation

# %%
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from HCP_MotterLai.model import (
    load_sc_and_loads, motter_lai, sweep_alpha, compute_per_node_vulnerability,
    classify_failure_mode, compute_weighted_vs_unweighted,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

np.random.seed(42)
# %matplotlib inline
print('Imports successful.')

# %% — Load data
sc_ctx, sc_ctx_labels, N, A_bin, G_w, G_uw, L0_uw_arr, L0_w_arr, degree_arr = load_sc_and_loads()

print(f'Network: {N} nodes')
print(f'Unweighted load — min: {L0_uw_arr.min():.1f}, max: {L0_uw_arr.max():.1f}, mean: {L0_uw_arr.mean():.1f}')
print(f'Weighted load   — min: {L0_w_arr.min():.1f},  max: {L0_w_arr.max():.1f}, mean: {L0_w_arr.mean():.1f}')

# Sanity check
g, nf = motter_lai(sc_ctx, attack_node=0, alpha=0.2)
print(f'Sanity check (node 0 removed, α=0.2): G = {g:.4f}, failed = {nf}')

# %% — Section 2: G vs α curves
load_arr = L0_uw_arr
node_max_degree = int(np.argmax(degree_arr))
node_max_load = int(np.argmax(load_arr))

print(f'Degree-based attack target : {sc_ctx_labels[node_max_degree]} (degree={degree_arr[node_max_degree]})')
print(f'Load-based attack target   : {sc_ctx_labels[node_max_load]}   (L0={load_arr[node_max_load]:.1f})')

alphas = np.linspace(0, 1.0, 30)
print('Sweeping α (this may take ~1 minute)...')
alphas, G_load, G_degree, G_rand = sweep_alpha(sc_ctx, node_max_load, node_max_degree, N, alphas)
print('Done.')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(alphas, G_load, 'o-', color='tomato', linewidth=2, markersize=5,
             label=f'Load-based attack ({sc_ctx_labels[node_max_load]})')
axes[0].plot(alphas, G_degree, 's-', color='darkorange', linewidth=2, markersize=5,
             label=f'Degree-based attack ({sc_ctx_labels[node_max_degree]})')
axes[0].plot(alphas, G_rand, '^-', color='steelblue', linewidth=2, markersize=5,
             label=f'Random attack (avg n=20)')
axes[0].axhline(1.0, color='k', linewidth=0.8, linestyle='--', alpha=0.5, label='No damage (G=1)')
axes[0].set_xlabel('Tolerance parameter α', fontsize=12)
axes[0].set_ylabel('G = N\'/N  (fraction in LCC)', fontsize=12)
axes[0].set_title('Motter–Lai: Network Damage vs Tolerance\n(HCP Cortical SC)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].set_xlim(0, 1.0)
axes[0].set_ylim(0, 1.05)
axes[0].grid(True, alpha=0.3)

axes[1].plot(alphas, 1 - np.array(G_load), 'o-', color='tomato', linewidth=2, markersize=5, label='Load-based')
axes[1].plot(alphas, 1 - np.array(G_degree), 's-', color='darkorange', linewidth=2, markersize=5, label='Degree-based')
axes[1].plot(alphas, 1 - np.array(G_rand), '^-', color='steelblue', linewidth=2, markersize=5, label='Random')
axes[1].axhline(0.0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
crit_alpha = alphas[next((i for i, g in enumerate(G_load) if g > 0.95), -1)]
axes[1].axvline(crit_alpha, color='tomato', linewidth=1.5, linestyle=':', alpha=0.7,
                label=f'Critical α ≈ {crit_alpha:.2f}')
axes[1].set_xlabel('Tolerance parameter α', fontsize=12)
axes[1].set_ylabel('Cascade damage = 1 - G', fontsize=12)
axes[1].set_title('Cascade Damage vs Tolerance', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].set_xlim(0, 1.0)
axes[1].set_ylim(-0.02, 1.0)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Motter–Lai Cascade Simulation on HCP Brain Network', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_motterlai_g_vs_alpha.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f'\nCritical tolerance α* (load-based, G>0.95): {crit_alpha:.2f}')

# %% — Section 3: Per-Node Cascade Vulnerability Map
ALPHA_LOW  = 0.1
ALPHA_MED  = 0.3
ALPHA_HIGH = 0.6

print(f'Running per-node cascade for all {N} nodes at α = {ALPHA_LOW}, {ALPHA_MED}, {ALPHA_HIGH}...')
node_df = compute_per_node_vulnerability(sc_ctx, N, ALPHA_LOW, ALPHA_MED, ALPHA_HIGH, load_arr, degree_arr)
node_df['Region'] = sc_ctx_labels
print('Done.')

top_danger = node_df.nsmallest(10, f'G_α={ALPHA_LOW}')
print(f'\nTop 10 most dangerous nodes to remove (α={ALPHA_LOW}):')
print(top_danger[['Region', 'L0', 'Degree', f'G_α={ALPHA_LOW}', f'Failed_α={ALPHA_LOW}']].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, alpha_val in zip(axes, [ALPHA_LOW, ALPHA_MED, ALPHA_HIGH]):
    col = f'G_α={alpha_val}'
    sorted_df = node_df.sort_values(col)
    damage = 1 - sorted_df[col].values
    colors = plt.cm.RdYlGn(sorted_df[col].values)

    bars = ax.barh(sorted_df['Region'], damage, color=colors, edgecolor='none', height=0.8)
    ax.axvline(0.05, color='k', linewidth=1, linestyle='--', alpha=0.5, label='5% damage')
    ax.set_title(f'Cascade Damage (1−G) per Node\nα = {alpha_val}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cascade damage (1 − G)')
    ax.tick_params(axis='y', labelsize=6)
    ax.set_xlim(0, max(damage) * 1.1 + 0.01)

    for j, (_, row) in enumerate(sorted_df.head(3).iterrows()):
        dmg = 1 - row[col]
        ax.text(dmg + 0.002, N - 1 - j, f'{row["Region"]} ({dmg:.2f})',
                va='center', fontsize=7, color='darkred')

plt.suptitle('Per-Node Cascade Vulnerability (Motter–Lai)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_motterlai_vulnerability.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 4: Cascade History
top3_nodes = node_df.nsmallest(3, f'G_α={ALPHA_LOW}')['Node'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, node_idx in zip(axes, top3_nodes):
    G_m, n_failed, history = motter_lai(sc_ctx, node_idx, ALPHA_LOW, return_history=True)

    fail_round = np.full(N, -1, dtype=int)
    for rnd, failed_set in enumerate(history):
        for node in failed_set:
            fail_round[node] = rnd

    rounds = [r for r in range(len(history))]
    counts = [len(h) for h in history]
    cumulative = np.cumsum(counts)

    ax2 = ax.twinx()
    ax.bar(rounds, counts, color='tomato', alpha=0.7, edgecolor='k', linewidth=0.5, label='Failed this round')
    ax2.plot(rounds, cumulative / N * 100, 'o-', color='darkred', linewidth=2,
             markersize=6, label='Cumulative % failed')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Cumulative % nodes failed', color='darkred', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='darkred')

    ax.set_xlabel('Cascade round')
    ax.set_ylabel('Nodes failed per round')
    ax.set_title(
        f'Seed: {sc_ctx_labels[node_idx]}\n'
        f'G={G_m:.3f}  |  {n_failed} total failed  |  {len(history)-1} rounds',
        fontsize=10, fontweight='bold'
    )
    ax.set_xticks(rounds)

    for rnd, failed_set in enumerate(history[1:], start=1):
        names = [sc_ctx_labels[n] for n in sorted(failed_set)]
        print(f'  [{sc_ctx_labels[node_idx]}] Round {rnd}: {names}')
    print()

plt.suptitle(f'Cascade Round-by-Round History  (α = {ALPHA_LOW})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_motterlai_history.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 5: Weighted vs Unweighted
(node_max_load_w, node_max_load_uw, alphas_short,
 G_uw_load, G_w_load, G_per_node_uw, G_per_node_w, r) = compute_weighted_vs_unweighted(
    sc_ctx, N, L0_w_arr, L0_uw_arr
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(alphas_short, G_uw_load, 'o-', color='steelblue', linewidth=2,
             markersize=5, label=f'Unweighted (seed: {sc_ctx_labels[node_max_load_uw]})')
axes[0].plot(alphas_short, G_w_load, 's-', color='tomato', linewidth=2,
             markersize=5, label=f'Weighted (seed: {sc_ctx_labels[node_max_load_w]})')
axes[0].set_xlabel('Tolerance α', fontsize=12)
axes[0].set_ylabel('G = N\'/N', fontsize=12)
axes[0].set_title('G vs α: Weighted vs Unweighted\n(load-based attack)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 1.05)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(G_per_node_uw, G_per_node_w, c=load_arr, cmap='plasma',
                s=60, edgecolors='k', linewidths=0.3)
for i in range(N):
    if abs(G_per_node_uw[i] - G_per_node_w[i]) > 0.1:
        axes[1].annotate(sc_ctx_labels[i], (G_per_node_uw[i], G_per_node_w[i]),
                         fontsize=7, xytext=(4, 2), textcoords='offset points')
lims = [min(min(G_per_node_uw), min(G_per_node_w)) - 0.02,
        max(max(G_per_node_uw), max(G_per_node_w)) + 0.02]
axes[1].plot(lims, lims, 'k--', linewidth=1, label='Identity')
axes[1].set_xlabel('G (unweighted)', fontsize=12)
axes[1].set_ylabel('G (weighted)', fontsize=12)
axes[1].set_title(f'Per-Node Cascade G: Weighted vs Unweighted\n(α=0.2,  r={r:.3f})',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=9)

plt.colorbar(axes[1].collections[0], ax=axes[1], label='Initial load L₀')
plt.suptitle('Weighted vs Unweighted Motter–Lai', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_motterlai_weighted_vs_uw.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 6: Failure Modes
col_low  = f'G_α={ALPHA_LOW}'
col_med  = f'G_α={ALPHA_MED}'
col_high = f'G_α={ALPHA_HIGH}'

node_df['Failure Mode'] = node_df.apply(
    lambda r: classify_failure_mode(r[col_low], r[col_med], r[col_high]), axis=1
)

mode_colors = {
    'Catastrophic': 'darkred',
    'Cascading':    'tomato',
    'Contained':    'darkorange',
    'Negligible':   'steelblue',
}

print('Failure mode distribution:')
print(node_df['Failure Mode'].value_counts().to_string())

# %% — Summary dashboard
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax_a = fig.add_subplot(gs[0, 0])
for mode, color in mode_colors.items():
    sub = node_df[node_df['Failure Mode'] == mode]
    ax_a.scatter(sub['Degree'], sub['L0'], c=color, s=80,
                 label=mode, edgecolors='k', linewidths=0.3, alpha=0.85, zorder=3)
for _, row in node_df[node_df['Failure Mode'].isin(['Catastrophic', 'Cascading'])].iterrows():
    ax_a.annotate(row['Region'], (row['Degree'], row['L0']),
                  fontsize=7, xytext=(4, 2), textcoords='offset points')
ax_a.set_xlabel('Degree', fontsize=10)
ax_a.set_ylabel('Initial Load L₀ (betweenness)', fontsize=10)
ax_a.set_title('Failure Mode by\nDegree & Load', fontsize=11, fontweight='bold')
ax_a.legend(fontsize=8)

ax_b = fig.add_subplot(gs[0, 1])
counts_mode = node_df['Failure Mode'].value_counts()
bars = ax_b.bar(counts_mode.index, counts_mode.values,
                color=[mode_colors[m] for m in counts_mode.index],
                edgecolor='k', linewidth=0.5)
ax_b.bar_label(bars, fontsize=11)
ax_b.set_title('Failure Mode Distribution\nacross 68 Cortical Regions', fontsize=11, fontweight='bold')
ax_b.set_ylabel('Number of regions')

ax_c = fig.add_subplot(gs[0, 2])
alphas_fine = np.array([ALPHA_LOW, ALPHA_MED, ALPHA_HIGH])
for _, row in node_df.iterrows():
    g_vals = [row[col_low], row[col_med], row[col_high]]
    ax_c.plot(alphas_fine, g_vals, '-', color=mode_colors[row['Failure Mode']],
              alpha=0.4, linewidth=1)
for mode, color in mode_colors.items():
    sub = node_df[node_df['Failure Mode'] == mode]
    if len(sub) == 0:
        continue
    g_mean = [sub[col_low].mean(), sub[col_med].mean(), sub[col_high].mean()]
    ax_c.plot(alphas_fine, g_mean, 'o-', color=color, linewidth=3,
              markersize=8, label=f'{mode} (mean)', zorder=5)
ax_c.set_xlabel('Tolerance α', fontsize=10)
ax_c.set_ylabel('G = N\'/N', fontsize=10)
ax_c.set_title('G vs α by Failure Mode\n(thin=individual, thick=mean)', fontsize=11, fontweight='bold')
ax_c.legend(fontsize=8)
ax_c.set_ylim(0, 1.05)
ax_c.grid(True, alpha=0.3)

ax_d = fig.add_subplot(gs[1, 0])
for mode, color in mode_colors.items():
    sub = node_df[node_df['Failure Mode'] == mode]
    if len(sub) > 0:
        ax_d.hist(sub['L0'], bins=15, color=color, alpha=0.6, label=mode, edgecolor='white')
ax_d.set_xlabel('Initial load L₀', fontsize=10)
ax_d.set_ylabel('Count', fontsize=10)
ax_d.set_title('Load Distribution\nby Failure Mode', fontsize=11, fontweight='bold')
ax_d.legend(fontsize=8)

ax_e = fig.add_subplot(gs[1, 1:])
alpha_star = []
for _, row in node_df.iterrows():
    g_low_val, g_med_val, g_high_val = row[col_low], row[col_med], row[col_high]
    if g_low_val > 0.95:    alpha_star.append(0.0)
    elif g_med_val > 0.95:  alpha_star.append(ALPHA_LOW + 0.01)
    elif g_high_val > 0.95: alpha_star.append(ALPHA_MED + 0.01)
    else:                    alpha_star.append(ALPHA_HIGH + 0.01)

node_df['alpha_star'] = alpha_star
sorted_df = node_df.sort_values('alpha_star', ascending=False)
bar_colors_e = [mode_colors[m] for m in sorted_df['Failure Mode']]
ax_e.barh(sorted_df['Region'], sorted_df['alpha_star'],
          color=bar_colors_e, edgecolor='none', height=0.7)
ax_e.axvline(ALPHA_LOW, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax_e.axvline(ALPHA_MED, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax_e.axvline(ALPHA_HIGH, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax_e.set_xlabel('Minimum α* for safe operation (G > 0.95)', fontsize=10)
ax_e.set_title('Critical Tolerance α* per Region\n(higher = more dangerous to remove)',
               fontsize=11, fontweight='bold')
ax_e.tick_params(axis='y', labelsize=6)

plt.suptitle('Motter–Lai Failure Mode Analysis — HCP Brain Network', fontsize=15, fontweight='bold')
plt.savefig(os.path.join(FIGS_DIR, 'hcp_motterlai_failure_modes.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f'\nMost dangerous region (highest α*): {node_df.loc[node_df["alpha_star"].idxmax(), "Region"]}')
print(f'Most benign region   (lowest  α*): {node_df.loc[node_df["alpha_star"].idxmin(), "Region"]}')
print(f'\nWeighted vs Unweighted Spearman r(G_uw, G_w) at α=0.2: {r:.3f}')
