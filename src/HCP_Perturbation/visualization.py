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
# # HCP Brain Network — Node Perturbation Analysis

# %%
import os, sys
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_TYPE', 'OSMesa')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from HCP_Perturbation.model import (
    load_sc_and_centrality, network_metrics, run_removal_simulations,
    compute_single_node_removal_impact, build_addition_strategies, cascade_failure,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(_DIR, '..', '..', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

# %matplotlib inline
print('Imports successful.')

# %% — Load data
(sc_ctx, sc_ctx_labels, n, A_bin, G_w, G_uw,
 strength, degree_bin, betweenness, eigenvec, closeness,
 hub_nodes, broker_nodes, bridge_nodes) = load_sc_and_centrality()

print(f'Loaded SC: {n} cortical regions')
print(f'Hubs    ({len(hub_nodes)}): {[sc_ctx_labels[i] for i in hub_nodes]}')
print(f'Brokers ({len(broker_nodes)}): {[sc_ctx_labels[i] for i in broker_nodes]}')
print(f'Bridges ({len(bridge_nodes)}): {[sc_ctx_labels[i] for i in bridge_nodes]}')

baseline = network_metrics(sc_ctx)
print('── Baseline Network Metrics ─────────────────────')
for k, v in baseline.items():
    print(f'  {k:<20}: {v:.4f}' if isinstance(v, float) else f'  {k:<20}: {v}')

# %% — Section 1: Node Removal
print('Running removal simulations...')
df_hub, df_broker, df_bridge, df_rand = run_removal_simulations(
    sc_ctx, n, strength, betweenness, bridge_nodes
)
print('Done.')

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

strategies = [
    (df_hub,    'Hub targeted',    'tomato'),
    (df_broker, 'Broker targeted', 'darkorange'),
    (df_bridge, 'Bridge targeted', 'steelblue'),
    (df_rand,   'Random (avg×20)', 'gray'),
]

metric_plot = [
    ('lcc_size',       'Largest Connected Component\n(fraction of original)', axes[0, 0]),
    ('global_eff',     'Global Efficiency',                                   axes[0, 1]),
    ('lambda1',        'Spectral Radius λ₁\n(↓ = lower epidemic risk)',       axes[1, 0]),
    ('avg_clustering', 'Average Clustering Coefficient',                      axes[1, 1]),
]

for metric, ylabel, ax in metric_plot:
    for df, label, color in strategies:
        if metric in df.columns:
            ax.plot(df['frac_removed'], df[metric], '-', color=color,
                    linewidth=2, label=label, alpha=0.85)
    ax.set_xlabel('Fraction of nodes removed')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel.split('\n')[0], fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.axhline(baseline.get(metric, np.nan), color='k', linewidth=0.8,
               linestyle=':', alpha=0.5, label='Baseline')

plt.suptitle('Node Removal Simulation — Targeted vs Random Attack', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_removal_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

print('\nFraction removed to reach LCC < 0.5:')
for df, label, _ in strategies:
    below = df[df['lcc_size'] < 0.5]
    frac = below['frac_removed'].iloc[0] if len(below) > 0 else '>1.0'
    print(f'  {label:<22}: {frac}')

# %% — Single node removal impact
delta_df = compute_single_node_removal_impact(
    sc_ctx, sc_ctx_labels, n, baseline,
    hub_nodes, broker_nodes, bridge_nodes, strength, betweenness
)

print('Impact of single-node removal (sorted by Global Efficiency drop):')
print(delta_df[['Region', 'Role', 'ΔLCC', 'ΔGlobal_Eff', 'Δλ₁']].to_string(index=False))

role_colors = {'Hub': 'tomato', 'Broker': 'darkorange', 'Bridge': 'steelblue'}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

delta_sorted = delta_df.sort_values('ΔGlobal_Eff')
colors = [role_colors[r] for r in delta_sorted['Role']]

axes[0].barh(delta_sorted['Region'], delta_sorted['ΔGlobal_Eff'],
             color=colors, edgecolor='k', linewidth=0.4)
axes[0].axvline(0, color='k', linewidth=0.8)
axes[0].set_title('ΔGlobal Efficiency after Single Node Removal\n(negative = network degraded)',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('ΔGlobal Efficiency')
for role, color in role_colors.items():
    axes[0].barh([], [], color=color, label=role)
axes[0].legend(fontsize=10)

axes[1].barh(delta_sorted['Region'], delta_sorted['Δλ₁'],
             color=colors, edgecolor='k', linewidth=0.4)
axes[1].axvline(0, color='k', linewidth=0.8)
axes[1].set_title('Δλ₁ after Single Node Removal\n(negative = epidemic risk reduced)',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Δ Spectral Radius λ₁')

plt.suptitle('Single Node Removal Impact on Key Regions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_single_removal_impact.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 2: Node Addition
additions = build_addition_strategies(sc_ctx, n, hub_nodes, strength, degree_bin, G_uw)

add_records = []
for name, A_new, w_new in additions:
    m = network_metrics(A_new)
    add_records.append({
        'Strategy':     name,
        'n_connections': int((w_new > 0).sum()),
        'ΔGlobal_Eff':  m['global_eff'] - baseline['global_eff'],
        'Δλ₁':          m['lambda1'] - baseline['lambda1'],
        'ΔClustering':  m['avg_clustering'] - baseline['avg_clustering'],
        'ΔLCC':         m['lcc_size'] - 1.0,
        'Δ_density':    m['density'] - baseline['density'],
        'Global_Eff':   m['global_eff'],
        'lambda1':      m['lambda1'],
    })

add_df = pd.DataFrame(add_records)
print(add_df[['Strategy', 'n_connections', 'ΔGlobal_Eff', 'Δλ₁', 'ΔClustering']].to_string(index=False))

add_colors = {'Hub-like': 'tomato', 'Bridge-type': 'steelblue',
              'Random': 'gray', 'Peripheral': 'khaki'}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics_add = [
    ('ΔGlobal_Eff', 'ΔGlobal Efficiency\n(↑ = better integration)'),
    ('Δλ₁',        'Δ Spectral Radius λ₁\n(↑ = higher epidemic risk)'),
    ('ΔClustering', 'Δ Clustering Coefficient\n(↑ = more local cohesion)'),
]

for ax, (col, ylabel) in zip(axes, metrics_add):
    bars = ax.bar(add_df['Strategy'], add_df[col],
                  color=[add_colors[s] for s in add_df['Strategy']],
                  edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.bar_label(bars, fmt='%.4f', fontsize=9, padding=2)
    ax.set_title(ylabel.split('\n')[0], fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel.split('\n')[1])
    ax.tick_params(axis='x', rotation=15)

plt.suptitle('Node Addition — Impact by Strategy', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_node_addition.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Connection patterns
def role_colors_fn(i):
    if i in hub_nodes:    return 'tomato'
    if i in broker_nodes: return 'darkorange'
    if i in bridge_nodes: return 'steelblue'
    return 'mediumpurple'

short_labels = [l.replace('L_', '').replace('R_', '') for l in sc_ctx_labels]

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.flatten()

for ax, (name, _, w_new) in zip(axes, additions):
    connected = np.where(w_new > 0)[0]
    ax.bar(range(n), w_new, color='lightgray', edgecolor='none')
    ax.bar(connected, w_new[connected],
           color=[role_colors_fn(i) for i in connected], edgecolor='k', linewidth=0.3)
    ax.set_xticks(connected)
    ax.set_xticklabels([short_labels[i] for i in connected], rotation=45, ha='right', fontsize=8)
    ax.set_title(f'{name} Addition ({len(connected)} connections)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Edge weight')

legend_handles = [Patch(color='tomato', label='Hub'),
                  Patch(color='darkorange', label='Broker'),
                  Patch(color='steelblue', label='Bridge'),
                  Patch(color='mediumpurple', label='Peripheral')]
fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=10, frameon=True)

plt.suptitle('Connection Patterns of Added Nodes', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(FIGS_DIR, 'hcp_addition_patterns.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 3: Cascade Failure
k_min = 3

seeds = [
    ('Hub',    hub_nodes[np.argmax(strength[hub_nodes])]),
    ('Broker', broker_nodes[np.argmax(betweenness[broker_nodes])] if len(broker_nodes) > 0 else hub_nodes[1]),
    ('Bridge', bridge_nodes[np.argmax(betweenness[bridge_nodes])] if len(bridge_nodes) > 0 else hub_nodes[2]),
    ('Random', np.random.randint(0, n)),
]

cascade_records = []
for role, seed in seeds:
    removed, active, m = cascade_failure(sc_ctx, seed, k_min=k_min)
    cascade_records.append({
        'Seed Region':     sc_ctx_labels[seed],
        'Seed Role':       role,
        'Total Removed':   len(removed),
        'Cascade Size':    len(removed) - 1,
        'Remaining Nodes': len(active),
        '% Network Lost':  len(removed) / n * 100,
        'Final LCC':       m.get('lcc_size', 0),
        'Final Global Eff': m.get('global_eff', 0),
        'Removed Regions': [sc_ctx_labels[r] for r in removed[1:]],
    })

cascade_df = pd.DataFrame(cascade_records)
print(f'Cascade failure simulation (k_min = {k_min})')
print(cascade_df[['Seed Region', 'Seed Role', 'Total Removed', 'Cascade Size', '% Network Lost', 'Final LCC']].to_string(index=False))

# %% — All-node cascade
all_cascade = []
for i in range(n):
    removed, active, _ = cascade_failure(sc_ctx, i, k_min=k_min)
    role = ('Hub' if i in hub_nodes else
            'Broker' if i in broker_nodes else
            'Bridge' if i in bridge_nodes else 'Peripheral')
    all_cascade.append({
        'node': i, 'region': sc_ctx_labels[i], 'role': role,
        'cascade_size': len(removed) - 1,
        'strength': strength[i], 'betweenness': betweenness[i],
    })
cascade_all_df = pd.DataFrame(all_cascade)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for role, color in role_colors.items():
    sub = cascade_all_df[cascade_all_df['role'] == role]
    axes[0].scatter(sub['strength'], sub['cascade_size'], c=color, s=70,
                    label=role, edgecolors='k', linewidths=0.3, alpha=0.85)
for _, row in cascade_all_df[cascade_all_df['cascade_size'] > 0].iterrows():
    axes[0].annotate(row['region'], (row['strength'], row['cascade_size']),
                     fontsize=7, xytext=(3, 2), textcoords='offset points')
axes[0].set_xlabel('Node Weighted Strength')
axes[0].set_ylabel('Cascade Size (secondary failures)')
axes[0].set_title(f'Cascade Failure Size vs Node Strength\n(k_min = {k_min})', fontsize=13, fontweight='bold')
axes[0].legend()

key_cascade = cascade_all_df[cascade_all_df['role'] != 'Peripheral'].sort_values('cascade_size', ascending=False)
colors_bar = [role_colors[r] for r in key_cascade['role']]
axes[1].bar(range(len(key_cascade)), key_cascade['cascade_size'], color=colors_bar, edgecolor='k', linewidth=0.4)
axes[1].set_xticks(range(len(key_cascade)))
axes[1].set_xticklabels(key_cascade['region'], rotation=45, ha='right', fontsize=8)
axes[1].set_title('Cascade Failure Size — Key Nodes Only', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Number of secondary node failures')
for role, color in role_colors.items():
    axes[1].bar([], [], color=color, label=role)
axes[1].legend()

plt.suptitle('Cascade Failure Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_cascade_failure.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% — Section 4: Before vs After Summary
worst_removal = delta_df.loc[delta_df['ΔGlobal_Eff'].idxmin()]
best_addition = add_df.loc[add_df['ΔGlobal_Eff'].idxmax()]

print(f'Most damaging removal : {worst_removal["Region"]} ({worst_removal["Role"]})')
print(f'  ΔGlobal Eff = {worst_removal["ΔGlobal_Eff"]:.5f}')
print(f'Most beneficial addition: {best_addition["Strategy"]} node')
print(f'  ΔGlobal Eff = {best_addition["ΔGlobal_Eff"]:.5f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metric_labels = ['Global Eff', 'λ₁', 'Avg Clustering']
baseline_vals = [baseline['global_eff'], baseline['lambda1'], baseline['avg_clustering']]

remaining_wr = [i for i in range(n) if sc_ctx_labels[i] != worst_removal['Region']]
sub_wr = sc_ctx[np.ix_(remaining_wr, remaining_wr)]
m_wr = network_metrics(sub_wr)
removal_vals = [m_wr['global_eff'], m_wr['lambda1'], m_wr['avg_clustering']]

A_best = dict(zip([s for s, _, _ in additions], [A for _, A, _ in additions]))[best_addition['Strategy']]
m_ba = network_metrics(A_best)
addition_vals = [m_ba['global_eff'], m_ba['lambda1'], m_ba['avg_clustering']]

x = np.arange(len(metric_labels))
w = 0.25
axes[0].bar(x - w, baseline_vals, w, label='Baseline', color='silver', edgecolor='k')
axes[0].bar(x, removal_vals, w, label=f'After {worst_removal["Region"]} removed', color='tomato', edgecolor='k')
axes[0].bar(x + w, addition_vals, w, label=f'After {best_addition["Strategy"]} added', color='steelblue', edgecolor='k')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metric_labels)
axes[0].set_title('Metric Comparison\nBaseline vs Removal vs Addition', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8)

rob_data = {
    'Hub\ntargeted':    np.trapz(df_hub['lcc_size'],    df_hub['frac_removed']),
    'Broker\ntargeted': np.trapz(df_broker['lcc_size'], df_broker['frac_removed']),
    'Bridge\ntargeted': np.trapz(df_bridge['lcc_size'], df_bridge['frac_removed']),
    'Random':           np.trapz(df_rand['lcc_size'],   df_rand['frac_removed']),
}
axes[1].bar(rob_data.keys(), rob_data.values(),
            color=['tomato', 'darkorange', 'steelblue', 'gray'], edgecolor='k')
axes[1].set_title('Robustness Index (AUC of LCC curve)\nHigher = more resilient to that attack', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Robustness Index')
for i, (k, v) in enumerate(rob_data.items()):
    axes[1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=10)

axes[2].scatter(add_df['ΔGlobal_Eff'], add_df['Δλ₁'],
                c=[add_colors[s] for s in add_df['Strategy']], s=200,
                edgecolors='k', linewidths=0.8, zorder=3)
for _, row in add_df.iterrows():
    axes[2].annotate(row['Strategy'], (row['ΔGlobal_Eff'], row['Δλ₁']),
                     fontsize=9, xytext=(5, 3), textcoords='offset points')
axes[2].axhline(0, color='k', linewidth=0.8, linestyle='--')
axes[2].axvline(0, color='k', linewidth=0.8, linestyle='--')
axes[2].set_xlabel('ΔGlobal Efficiency (↑ = better)')
axes[2].set_ylabel('Δλ₁ (↑ = higher epidemic risk)')
axes[2].set_title('Addition Strategy:\nEfficiency vs Epidemic Risk Tradeoff', fontsize=12, fontweight='bold')
axes[2].fill_between([-1, 0], [0, 0], [1, 1], alpha=0.07, color='tomato')
axes[2].fill_between([0, 1], [0, 0], [1, 1], alpha=0.07, color='green')
axes[2].fill_between([0, 1], [-1, -1], [0, 0], alpha=0.12, color='steelblue')

plt.suptitle('Perturbation Analysis — Final Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'hcp_perturbation_summary.png'), dpi=150, bbox_inches='tight')
plt.show()
