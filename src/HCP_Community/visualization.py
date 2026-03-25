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
# # HCP Brain Network — Community Detection & Functional Labelling

# %%
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from HCP_Community.model import (
    load_sc_with_networks, run_all_algorithms, build_summary_df, compute_nmi_matrix,
    label_communities, participation_coefficient, intra_z_score, cartographic_role,
    compute_between_community_strength, NETWORK_COLORS,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

np.random.seed(42)
# %matplotlib inline
print('All imports successful.')

# %% — Load data
sc_ctx, sc_ctx_labels, n, A_bin, G_w, G_uw, short_labels, known_networks = load_sc_with_networks()
print(f'Loaded {n} cortical regions')

# %% — Section 1: Run all algorithms
print('Running community detection algorithms...')
all_labels, all_Q = run_all_algorithms(sc_ctx, n, G_w, G_uw)
print('Done.')

for name, labs in all_labels.items():
    print(f'{name}: {len(np.unique(labs))} communities, Q = {all_Q[name]:.4f}')

summary_df = build_summary_df(all_labels, all_Q)
print(summary_df.to_string(index=False))

nmi_matrix, alg_names = compute_nmi_matrix(all_labels)
nmi_df = pd.DataFrame(nmi_matrix, index=alg_names, columns=alg_names)
print('\nNormalized Mutual Information between algorithm pairs:')
print(nmi_df.round(3).to_string())

# %% — NMI heatmap + modularity bar
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

im = axes[0].imshow(nmi_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(alg_names)))
axes[0].set_yticks(range(len(alg_names)))
axes[0].set_xticklabels(alg_names, rotation=30, ha='right', fontsize=9)
axes[0].set_yticklabels(alg_names, fontsize=9)
for i in range(len(alg_names)):
    for j in range(len(alg_names)):
        axes[0].text(j, i, f'{nmi_matrix[i,j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=axes[0], label='NMI')
axes[0].set_title('Algorithm Agreement (NMI)\n1.0 = identical partitions', fontsize=13, fontweight='bold')

colors_bar = ['#E05C5C', '#4A90D9', '#27AE60', '#F39C12', '#8E44AD']
bars = axes[1].bar(alg_names, [all_Q[a] for a in alg_names],
                   color=colors_bar, edgecolor='k', linewidth=0.5)
axes[1].bar_label(bars, fmt='%.4f', fontsize=10, padding=2)
axes[1].set_title('Modularity Q by Algorithm\n(higher = stronger community structure)',
                  fontsize=13, fontweight='bold')
axes[1].set_ylabel('Modularity Q')
axes[1].set_ylim(0, max(all_Q.values()) * 1.15)
axes[1].tick_params(axis='x', rotation=20)

plt.suptitle('Community Detection Algorithm Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_community_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 2: Functional Labelling
btw_arr = np.array(list(nx.betweenness_centrality(G_uw, normalized=True).values()))
strength_arr = sc_ctx.sum(axis=1)

all_labelled = pd.concat([
    label_communities(labs, name, sc_ctx_labels, known_networks, strength_arr, btw_arr)
    for name, labs in all_labels.items()
], ignore_index=True)

leiden_labelled = all_labelled[all_labelled['Method'] == 'Leiden'].copy()
labels_leiden = all_labels['Leiden']

print('Leiden community functional labels:')
print(leiden_labelled[['Community ID', 'Size', 'Functional Label', 'Purity', 'Avg Strength', 'Avg Betweenness']].to_string(index=False))

# %% — Leiden adjacency matrix visualization
sort_order = np.lexsort((strength_arr, labels_leiden))
sc_sorted = sc_ctx[np.ix_(sort_order, sort_order)]

_, counts = np.unique(labels_leiden[sort_order], return_counts=True)
boundaries = np.cumsum(counts)[:-1] - 0.5

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

im = axes[0].imshow(np.log1p(sc_sorted), cmap='inferno', aspect='auto')
for b in boundaries:
    axes[0].axhline(b, color='cyan', linewidth=1.0)
    axes[0].axvline(b, color='cyan', linewidth=1.0)
plt.colorbar(im, ax=axes[0], label='log(streamlines + 1)')
axes[0].set_title('Adjacency Matrix — Leiden Communities\n(sorted by community)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Region (sorted)')
axes[0].set_ylabel('Region (sorted)')

fn_labels = leiden_labelled['Functional Label'].tolist()
fn_colors = [NETWORK_COLORS.get(l, '#BDC3C7') for l in fn_labels]
sizes = leiden_labelled['Size'].tolist()
purities = leiden_labelled['Purity'].tolist()
comm_ids = leiden_labelled['Community ID'].tolist()

bars = axes[1].barh([f'C{i}' for i in comm_ids], sizes, color=fn_colors, edgecolor='k', linewidth=0.4)
for bar, purity, label in zip(bars, purities, fn_labels):
    axes[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                 f'{label}  ({purity:.0%})', va='center', fontsize=8)
axes[1].set_xlabel('Number of regions')
axes[1].set_title('Leiden Community Size & Functional Label\n(colour = dominant network)', fontsize=13, fontweight='bold')
axes[1].set_xlim(0, max(sizes) + 22)

handles = [mpatches.Patch(color=c, label=nname) for nname, c in NETWORK_COLORS.items()]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, frameon=True)

plt.suptitle('Leiden Community Detection — Functional Labelling', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(os.path.join(FIGS_DIR, 'hcp_leiden_communities.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 3: Cross-Algorithm Functional Label Consistency
region_label_df = pd.DataFrame({'Region': sc_ctx_labels, 'Known Network': known_networks})

for name, labs in all_labels.items():
    labelled = label_communities(labs, name, sc_ctx_labels, known_networks, strength_arr, btw_arr)
    node_to_funclabel = {}
    for _, row in labelled.iterrows():
        for member in row['Members']:
            node_to_funclabel[member] = row['Functional Label']
    region_label_df[name] = [node_to_funclabel.get(r, 'Unassigned') for r in sc_ctx_labels]

algo_cols = list(all_labels.keys())
region_label_df['Consensus'] = region_label_df[algo_cols].apply(
    lambda row: row.value_counts().index[0], axis=1
)
region_label_df['Agreement'] = region_label_df[algo_cols].apply(
    lambda row: row.value_counts().iloc[0] / len(algo_cols), axis=1
)

print('Per-region consensus functional labels (sorted by agreement):')
print(region_label_df[['Region', 'Known Network', 'Consensus', 'Agreement']]
      .sort_values('Agreement')
      .to_string(index=False))

# %% — Agreement heatmap
all_networks = list(NETWORK_COLORS.keys())
net_to_int = {net: i for i, net in enumerate(all_networks)}

heat_data = region_label_df[algo_cols].applymap(
    lambda x: net_to_int.get(x, len(all_networks) - 1)
).values

sort_idx = region_label_df.sort_values(['Consensus', 'Agreement'], ascending=[True, False]).index

fig, axes = plt.subplots(1, 2, figsize=(18, 14))

cmap = plt.cm.get_cmap('tab10', len(all_networks))
im = axes[0].imshow(heat_data[sort_idx], cmap=cmap, aspect='auto',
                    vmin=0, vmax=len(all_networks) - 1)
axes[0].set_xticks(range(len(algo_cols)))
axes[0].set_xticklabels(algo_cols, rotation=30, ha='right', fontsize=9)
axes[0].set_yticks(range(n))
axes[0].set_yticklabels(region_label_df.loc[sort_idx, 'Region'], fontsize=6)
axes[0].set_title('Functional Label per Region × Algorithm\n(rows sorted by consensus label)',
                  fontsize=12, fontweight='bold')
handles = [mpatches.Patch(color=cmap(i), label=nnet) for i, nnet in enumerate(all_networks)]
axes[0].legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.9)

agreement_sorted = region_label_df.loc[sort_idx, 'Agreement']
consensus_sorted = region_label_df.loc[sort_idx, 'Consensus']
bar_colors = [NETWORK_COLORS.get(c, '#BDC3C7') for c in consensus_sorted]

axes[1].barh(range(n), agreement_sorted.values, color=bar_colors,
             edgecolor='none', height=0.85)
axes[1].set_yticks(range(n))
axes[1].set_yticklabels(region_label_df.loc[sort_idx, 'Region'], fontsize=6)
axes[1].axvline(0.6, color='k', linewidth=1, linestyle='--', label='60% agreement')
axes[1].set_xlabel('Algorithm agreement (fraction)')
axes[1].set_title('Cross-Algorithm Agreement per Region\n(colour = consensus label)',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)

plt.suptitle('Cross-Algorithm Functional Label Consistency', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_cross_algorithm_labels.png'), dpi=150, bbox_inches='tight')
plt.show()

low_agree = region_label_df[region_label_df['Agreement'] < 0.6]
print(f'\nRegions with <60% cross-algorithm agreement ({len(low_agree)}):')
print(low_agree[['Region', 'Known Network', 'Consensus', 'Agreement']].to_string(index=False))

# %% — Section 4: Community-Level Network Properties
within_density, within_strength, between_strength, unique_comms, n_comm = compute_between_community_strength(
    sc_ctx, labels_leiden
)

pc = participation_coefficient(sc_ctx, labels_leiden)
z_score = intra_z_score(sc_ctx, labels_leiden)
cartographic_roles = cartographic_role(z_score, pc)

cart_df = pd.DataFrame({
    'Region':    sc_ctx_labels,
    'Community': labels_leiden,
    'Func Label': [known_networks[i] for i in range(n)],
    'z (intra)': z_score.round(3),
    'PC':        pc.round(3),
    'Cart Role': cartographic_roles,
})

print('Cartographic role distribution:')
print(cart_df['Cart Role'].value_counts().to_string())

# %% — Cartographic map
cart_colors = {
    'Provincial Hub':  'darkred',
    'Connector Hub':   'tomato',
    'Kinless Hub':     'salmon',
    'Ultra-peripheral': 'lightgray',
    'Peripheral':      'steelblue',
    'Connector':       'darkorange',
    'Kinless':         'mediumpurple',
}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for role, color in cart_colors.items():
    sub = cart_df[cart_df['Cart Role'] == role]
    if len(sub) == 0:
        continue
    axes[0].scatter(sub['PC'], sub['z (intra)'], c=color, s=70,
                    label=role, edgecolors='k', linewidths=0.3, alpha=0.85)
for _, row in cart_df[cart_df['Cart Role'].str.contains('Hub')].iterrows():
    axes[0].annotate(row['Region'], (row['PC'], row['z (intra)']),
                     fontsize=7, xytext=(4, 2), textcoords='offset points')

axes[0].axhline(2.5, color='k', linewidth=1.2, linestyle='--', label='Hub threshold (z=2.5)')
axes[0].axvline(0.62, color='k', linewidth=1.0, linestyle=':', alpha=0.6)
axes[0].axvline(0.80, color='k', linewidth=1.0, linestyle=':', alpha=0.6)
axes[0].set_xlabel('Participation Coefficient (PC)', fontsize=11)
axes[0].set_ylabel('Within-community z-score', fontsize=11)
axes[0].set_title('Cartographic Map\n(Guimerà & Amaral 2005)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=8, loc='upper left')

fn_labels_comm = [leiden_labelled[leiden_labelled['Community ID'] == c]['Functional Label'].values[0]
                  for c in unique_comms]
bw_labels = [f'C{c}\n{fn_labels_comm[i][:12]}' for i, c in enumerate(unique_comms)]
im2 = axes[1].imshow(np.log1p(between_strength), cmap='Blues', aspect='auto')
axes[1].set_xticks(range(n_comm))
axes[1].set_yticks(range(n_comm))
axes[1].set_xticklabels(bw_labels, rotation=30, ha='right', fontsize=8)
axes[1].set_yticklabels(bw_labels, fontsize=8)
for i in range(n_comm):
    for j in range(n_comm):
        if i != j:
            axes[1].text(j, i, f'{between_strength[i,j]:.0f}', ha='center',
                         va='center', fontsize=7)
plt.colorbar(im2, ax=axes[1], label='log(inter-community strength + 1)')
axes[1].set_title('Between-Community Connectivity\n(Leiden)', fontsize=13, fontweight='bold')

plt.suptitle('Community Structure Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_community_structure.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Final community profile
final_rows = []
for _, row in leiden_labelled.iterrows():
    cid = row['Community ID']
    members = np.where(labels_leiden == cid)[0]
    hub_count = sum(1 for i in members if 'Hub' in cart_df.loc[cart_df['Region'] == sc_ctx_labels[i], 'Cart Role'].values[0])
    conn_count = sum(1 for i in members if 'Connector' in cart_df.loc[cart_df['Region'] == sc_ctx_labels[i], 'Cart Role'].values[0])
    final_rows.append({
        'Community':          f'C{cid}',
        'Functional Label':   row['Functional Label'],
        'Size':               row['Size'],
        'Purity':             f"{row['Purity']:.0%}",
        'Hubs inside':        hub_count,
        'Connectors inside':  conn_count,
        'Within density':     round(within_density[cid], 3),
        'Avg Strength':       row['Avg Strength'],
    })

final_df = pd.DataFrame(final_rows)
print('Final Community Functional Profile (Leiden):')
print(final_df.to_string(index=False))
