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
# # Alzheimer's Disease as a Brain Network Cascade — Comparison with Motter–Lai Simulation

# %%
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from tqdm.notebook import tqdm

from HCP_AlzheimerComparison.model import (
    load_sc_and_classify, global_efficiency, simulate_braak_removal,
    motter_lai, sweep_entorhinal_vs_hub, sweep_braak_multiseed,
    compute_vulnerability_map, BRAAK_KEYWORDS,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

np.random.seed(42)
# %matplotlib inline
print('Imports successful.')

# %% — Load data
sc_ctx, labels, N, braak_indices, dmn_indices, bet_arr, deg_arr, eig_arr = load_sc_and_classify()

for stage, kws in BRAAK_KEYWORDS.items():
    matched = [(i, labels[i]) for i in braak_indices[stage]]
    print(f"\nBraak {stage} ({len(matched)} regions):")
    for i, lab in matched:
        print(f"  [{i:2d}] {lab}")

E0 = global_efficiency(sc_ctx)
print(f'\nBaseline global efficiency: {E0:.4f}')
print(f'Top-5 betweenness nodes: {[labels[i] for i in np.argsort(bet_arr)[-5:][::-1]]}')

# %% — Section 1: Braak-Stage Sequential Attack
df_braak, eff_base = simulate_braak_removal(sc_ctx, braak_indices)
lcc_base = df_braak.iloc[0]['lcc']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

x = range(len(df_braak))
stage_labels = df_braak['stage'].str.replace(r'\s*\(.*\)', '', regex=True)
stage_colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

ax = axes[0]
bars = ax.bar(x, df_braak['lcc'], color=stage_colors[:len(df_braak)])
ax.axhline(lcc_base, color='gray', ls='--', lw=1, label='Baseline')
ax.set_xticks(list(x))
ax.set_xticklabels(stage_labels, rotation=15, ha='right', fontsize=8)
ax.set_ylabel('LCC fraction (G)')
ax.set_title('Braak-Stage Attack: Network Fragmentation')
ax.set_ylim(0, 1.05)
ax.legend()

ax = axes[1]
ax.bar(x, df_braak['eff'], color=stage_colors[:len(df_braak)])
ax.axhline(eff_base, color='gray', ls='--', lw=1, label='Baseline')
ax.set_xticks(list(x))
ax.set_xticklabels(stage_labels, rotation=15, ha='right', fontsize=8)
ax.set_ylabel('Global Efficiency')
ax.set_title('Braak-Stage Attack: Communication Efficiency')
ax.legend()

plt.suptitle('Simulated AD Progression — Braak Staging Sequence', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'ad_braak_progression.png'), dpi=150, bbox_inches='tight')
plt.show()

eff_drop = (eff_base - df_braak['eff'].iloc[-1]) / eff_base * 100
lcc_drop = (lcc_base - df_braak['lcc'].iloc[-1]) / lcc_base * 100
print(f'\nInterpretation:')
print(f'  Final LCC drop:        {lcc_drop:.1f}%  ({df_braak["n_removed"].iloc[-1]} of {N} nodes removed)')
print(f'  Global efficiency drop: {eff_drop:.1f}%')

# %% — Section 2: Hub Vulnerability Analysis
all_braak_all = set(braak_indices['I-II (Transentorhinal)'] +
                    braak_indices['III-IV (Limbic)'] +
                    braak_indices['V-VI (Isocortical)'])

colors_ad = []
ad_labels_list = []
for i in range(N):
    if i in braak_indices['I-II (Transentorhinal)']:
        colors_ad.append('#e74c3c'); ad_labels_list.append('Braak I-II')
    elif i in braak_indices['III-IV (Limbic)']:
        colors_ad.append('#e67e22'); ad_labels_list.append('Braak III-IV')
    elif i in braak_indices['V-VI (Isocortical)']:
        colors_ad.append('#f1c40f'); ad_labels_list.append('Braak V-VI')
    else:
        colors_ad.append('#95a5a6'); ad_labels_list.append('Not in Braak map')

patches = [
    mpatches.Patch(color='#e74c3c', label='Braak I-II (entorhinal)'),
    mpatches.Patch(color='#e67e22', label='Braak III-IV (limbic)'),
    mpatches.Patch(color='#f1c40f', label='Braak V-VI (isocortical)'),
    mpatches.Patch(color='#95a5a6', label='Unaffected'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(deg_arr, bet_arr, c=colors_ad, alpha=0.7, s=60, edgecolors='k', lw=0.4)
for i in list(braak_indices['I-II (Transentorhinal)']) + list(braak_indices['III-IV (Limbic)'])[:4]:
    ax.annotate(labels[i].replace('lh_', '').replace('rh_', ''),
                (deg_arr[i], bet_arr[i]), fontsize=7, ha='left',
                xytext=(3, 3), textcoords='offset points')
ax.legend(handles=patches, fontsize=8)
ax.set_xlabel('Weighted Degree (strength)')
ax.set_ylabel('Betweenness Centrality (unnormalised)')
ax.set_title('Hub Position vs AD Braak Stage')

ax = axes[1]
ax.scatter(eig_arr, bet_arr, c=colors_ad, alpha=0.7, s=60, edgecolors='k', lw=0.4)
for i in braak_indices['I-II (Transentorhinal)']:
    ax.annotate(labels[i].replace('lh_', '').replace('rh_', ''),
                (eig_arr[i], bet_arr[i]), fontsize=7, ha='left',
                xytext=(3, 3), textcoords='offset points')
ax.legend(handles=patches, fontsize=8)
ax.set_xlabel('Eigenvector Centrality')
ax.set_ylabel('Betweenness Centrality')
ax.set_title('Influence vs Bridging — AD Vulnerability')

plt.suptitle('Hub Vulnerability in AD (cf. Buckner et al. 2009)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'ad_hub_vulnerability.png'), dpi=150, bbox_inches='tight')
plt.show()

ad_mask = np.array([i in all_braak_all for i in range(N)])
bet_ad = bet_arr[ad_mask]
bet_non = bet_arr[~ad_mask]
stat, pval = stats.mannwhitneyu(bet_ad, bet_non, alternative='greater')
print(f'\nMann-Whitney U test (AD betweenness > non-AD):')
print(f'  Median betweenness — AD regions:     {np.median(bet_ad):.1f}')
print(f'  Median betweenness — other regions:  {np.median(bet_non):.1f}')
print(f'  U={stat:.0f}, p={pval:.4f} → {"SIGNIFICANT" if pval < 0.05 else "not significant"}')

# %% — Section 3: Motter-Lai Cascade from AD Epicentres
ALPHAS = np.linspace(0.0, 1.0, 20)

print('\nRunning Motter-Lai sweeps...')
results, entorhinal_idx, top_bet_node, all_braak_all2, non_ad_indices = sweep_entorhinal_vs_hub(
    sc_ctx, N, labels, braak_indices, bet_arr, ALPHAS
)
print('Done.')

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(ALPHAS, results['entorhinal_lh'], 'r-o', ms=5, lw=2,
        label='Entorhinal (LH) — AD epicentre (Braak I)')
ax.plot(ALPHAS, results['entorhinal_rh'], 'r--^', ms=5, lw=2,
        label='Entorhinal (RH) — AD epicentre (Braak I)')
ax.plot(ALPHAS, results['top_hub'], 'b-s', ms=5, lw=2,
        label=f'Top hub ({labels[top_bet_node]}) — highest betweenness')
ax.fill_between(ALPHAS,
                results['random_mean'] - results['random_std'],
                results['random_mean'] + results['random_std'],
                alpha=0.25, color='gray')
ax.plot(ALPHAS, results['random_mean'], 'k--', lw=1.5,
        label='Random attack (mean ± SD, non-AD regions)')

ax.set_xlabel('Tolerance parameter α', fontsize=12)
ax.set_ylabel('Surviving network fraction G = N\'/N', fontsize=12)
ax.set_title('Motter–Lai Cascade: Entorhinal (AD Epicentre) vs Hub vs Random Attack', fontsize=13)
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.axhline(0.5, color='salmon', ls=':', lw=1, label='50% threshold')

for key, col, lab in [('entorhinal_lh', 'red', 'Entorhinal LH'),
                       ('top_hub', 'blue', 'Top hub')]:
    alpha_crit = ALPHAS[np.where(results[key] > 0.9)[0][0]] if np.any(results[key] > 0.9) else None
    if alpha_crit:
        ax.axvline(alpha_crit, color=col, ls=':', alpha=0.5)
        ax.text(alpha_crit + 0.01, 0.15, f'α*={alpha_crit:.2f}\n({lab})', color=col, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'ad_motterlai_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 4: Multi-Seed Braak-Stage Cascade
ALPHA_TEST = [0.1, 0.3, 0.5, 1.0]
print('Running multi-seed Braak cascades...')
multi_results, ALPHA_TEST, STAGE_KEYS = sweep_braak_multiseed(sc_ctx, braak_indices, ALPHA_TEST)
print('Done.')

alpha_colors = {0.1: '#d62728', 0.3: '#ff7f0e', 0.5: '#2ca02c', 1.0: '#1f77b4'}

fig, ax = plt.subplots(figsize=(11, 5))

for alpha in ALPHA_TEST:
    rows = multi_results[alpha]
    x_pts = list(range(len(rows)))
    g_vals = [r['G'] for r in rows]
    ax.plot(x_pts, g_vals, 'o-', color=alpha_colors[alpha], lw=2, ms=8, label=f'α = {alpha}')
    for xi, row in enumerate(rows):
        ax.annotate(f"{row['total_lost']}",
                    (xi, row['G']), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color=alpha_colors[alpha])

ax.set_xticks(range(4))
ax.set_xticklabels(['Baseline', 'Braak I-II\n(Entorhinal)', 'Braak III-IV\n(Limbic)',
                    'Braak V-VI\n(Isocortical)'], fontsize=10)
ax.set_ylabel('G = Surviving LCC fraction (post-cascade)', fontsize=11)
ax.set_title('Multi-Seed Braak-Stage Cascade\n(numbers = total nodes lost incl. cascade)', fontsize=12)
ax.legend(title='Network reserve α', fontsize=10)
ax.set_ylim(0, 1.1)
ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5, label='50% threshold')

for xi, label_txt in [(1, 'Pre-clinical'), (2, 'MCI / Early AD'), (3, 'Moderate–Severe AD')]:
    ax.text(xi, 1.06, label_txt, ha='center', fontsize=8, style='italic', color='dimgray')

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'ad_braak_cascade.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 5: Per-Region AD Vulnerability Map
ALPHA_VUL = 0.2
print(f'Computing per-node G at α={ALPHA_VUL}...')
G_per_node, vulnerability_ml, bio_score, bio_score_continuous, r, p = compute_vulnerability_map(
    sc_ctx, N, bet_arr, braak_indices, ALPHA_VUL
)
print(f'Correlation (biological vs ML vulnerability): {r:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
sc_scatter = ax.scatter(bio_score_continuous, vulnerability_ml,
                c=colors_ad, s=60, alpha=0.75, edgecolors='k', lw=0.4)

top10_ml = np.argsort(vulnerability_ml)[-10:][::-1]
for i in top10_ml:
    short = labels[i].replace('lh_', 'L.').replace('rh_', 'R.')
    ax.annotate(short, (bio_score_continuous[i], vulnerability_ml[i]),
                fontsize=6.5, xytext=(4, 0), textcoords='offset points')

ax.set_xlabel('Biological AD Vulnerability\n(Braak stage weight + betweenness)')
ax.set_ylabel(f'Motter–Lai Vulnerability (1−G at α={ALPHA_VUL})')
ax.set_title(f'Biological vs Network-Cascade Vulnerability\n(Spearman r={r:.3f}, p={p:.3f})')
ax.legend(handles=patches, fontsize=8)

ax = axes[1]
top20 = np.argsort(vulnerability_ml)[-20:][::-1]
bar_colors = [colors_ad[i] for i in top20]
short_labs = [labels[i].replace('lh_', 'L.').replace('rh_', 'R.') for i in top20]
ax.barh(range(20), vulnerability_ml[top20][::-1],
        color=bar_colors[::-1], edgecolor='k', lw=0.4)
ax.set_yticks(range(20))
ax.set_yticklabels(short_labs[::-1], fontsize=8)
ax.set_xlabel(f'Cascade Vulnerability (1−G) at α={ALPHA_VUL}')
ax.set_title('Top 20 Most Vulnerable Nodes\n(Motter–Lai) — coloured by Braak stage')
ax.legend(handles=patches, fontsize=8, loc='lower right')

plt.suptitle('AD Biological vs Network-Cascade Vulnerability', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'ad_vulnerability_map.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 6: Comparison Table
non_ad_idx = [i for i in range(N) if i not in all_braak_all]
groups_vul = [
    vulnerability_ml[braak_indices['I-II (Transentorhinal)']],
    vulnerability_ml[braak_indices['III-IV (Limbic)']],
    vulnerability_ml[braak_indices['V-VI (Isocortical)']],
    vulnerability_ml[non_ad_idx],
]
f_stat, p_val = stats.f_oneway(*groups_vul)
print(f'\nOne-way ANOVA (ML vulnerability across Braak groups):')
print(f'  F = {f_stat:.3f}, p = {p_val:.4f} → {"SIGNIFICANT" if p_val < 0.05 else "not significant"}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

group_labels = ['Braak I-II\n(Entorhinal)', 'Braak III-IV\n(Limbic)',
                'Braak V-VI\n(Isocortical)', 'Non-AD\nRegions']
group_vul_means = [g.mean() for g in groups_vul]
group_vul_stds = [g.std() for g in groups_vul]
bar_cols = ['#e74c3c', '#e67e22', '#f1c40f', '#95a5a6']

ax = axes[0]
bars = ax.bar(group_labels, group_vul_means, yerr=group_vul_stds,
              color=bar_cols, capsize=5, edgecolor='k', lw=0.7)
ax.set_ylabel('Mean Cascade Vulnerability (1−G at α=0.2)')
ax.set_title(f'Motter–Lai Vulnerability by Braak Stage\n(ANOVA F={f_stat:.2f}, p={p_val:.3f})')
ax.set_ylim(0, max(group_vul_means) * 1.4)

ax = axes[1]
braak_eff_loss = [(eff_base - df_braak.iloc[i + 1]['eff']) / eff_base * 100
                  for i in range(3)]
ml_stage_damage = []
for stage_key in STAGE_KEYS:
    stage_nodes = braak_indices[stage_key]
    if len(stage_nodes) > 0:
        mean_vul = vulnerability_ml[stage_nodes].mean() * 100
        ml_stage_damage.append(mean_vul)
    else:
        ml_stage_damage.append(0)

x_pos = np.arange(3)
w = 0.35
bars1 = ax.bar(x_pos - w / 2, braak_eff_loss, w, color=['#e74c3c', '#e67e22', '#f1c40f'],
               label='Biological: % efficiency loss (cumulative)', edgecolor='k', lw=0.7)
bars2 = ax.bar(x_pos + w / 2, ml_stage_damage, w, color='#3498db', alpha=0.8,
               label='Motter–Lai: mean cascade damage (%) when seeded from stage', edgecolor='k', lw=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Braak I-II', 'Braak III-IV', 'Braak V-VI'])
ax.set_ylabel('Damage (%)')
ax.set_title('AD Biological Damage vs Motter–Lai Prediction\n(per Braak Stage)')
ax.legend(fontsize=8)

plt.suptitle('Alzheimer\'s Disease — Biological vs Network-Cascade Damage Comparison', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'ad_comparison_table.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Final summary
print('=' * 70)
print('  ALZHEIMER\'S DISEASE vs MOTTER-LAI — KEY FINDINGS SUMMARY')
print('=' * 70)
print('\n1. BRAAK-STAGE NETWORK DAMAGE')
for _, row in df_braak.iterrows():
    eff_pct = (eff_base - row['eff']) / eff_base * 100
    lcc_pct = (1 - row['lcc']) * 100
    print(f'   {row["stage"]:<35}  LCC={row["lcc"]:.3f} ({lcc_pct:.1f}% loss)  Eff_drop={eff_pct:.1f}%')
print(f'\n2. HUB VULNERABILITY')
print(f'   AD-affected regions median betweenness: {np.median(bet_ad):.1f}')
print(f'   Non-AD regions median betweenness:      {np.median(bet_non):.1f}')
print(f'   Mann-Whitney p={pval:.4f}')
print(f'\n3. SPEARMAN CORRELATION (biological AD vulnerability vs ML cascade)')
print(f'   r = {r:.3f} (p={p:.3f})')
