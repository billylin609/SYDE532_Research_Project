# %% model.py — HCP_Perturbation: data loading and perturbation simulations

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from copy import deepcopy

from enigmatoolbox.datasets import load_sc

np.random.seed(42)


def load_sc_and_centrality():
    """Load HCP SC and compute all centrality measures."""
    sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()
    n = sc_ctx.shape[0]
    A_bin = (sc_ctx > 0).astype(float)

    G_w = nx.from_numpy_array(sc_ctx)
    G_uw = nx.from_numpy_array(A_bin)

    G_dist = G_w.copy()
    for u, v, d in G_dist.edges(data=True):
        d['distance'] = 1.0 / (d['weight'] + 1e-9)

    strength = np.array([d for _, d in G_w.degree(weight='weight')])
    degree_bin = np.array([d for _, d in G_uw.degree()])
    betweenness = np.array(list(nx.betweenness_centrality(G_dist, weight='distance', normalized=True).values()))
    eigenvec = np.array(list(nx.eigenvector_centrality_numpy(G_w, weight='weight').values()))
    closeness = np.array(list(nx.closeness_centrality(G_dist, distance='distance').values()))

    z_str = stats.zscore(strength)
    z_btw = stats.zscore(betweenness)
    z_eig = stats.zscore(eigenvec)
    z_deg = stats.zscore(degree_bin)

    is_hub = (z_str > 1.0) & (z_eig > 1.0)
    is_broker = (z_btw > 1.0) & ~is_hub
    is_bridge = (z_btw > 0.5) & (z_deg < 0.0) & ~is_hub & ~is_broker

    hub_nodes = np.where(is_hub)[0]
    broker_nodes = np.where(is_broker)[0]
    bridge_nodes = np.where(is_bridge)[0]

    return (sc_ctx, sc_ctx_labels, n, A_bin, G_w, G_uw,
            strength, degree_bin, betweenness, eigenvec, closeness,
            hub_nodes, broker_nodes, bridge_nodes)


def network_metrics(A, labels=None):
    """Return dict of key metrics for a weighted adjacency matrix."""
    m = {}
    n_nodes = A.shape[0]
    A_b = (A > 0).astype(float)
    G = nx.from_numpy_array(A_b)
    Gw = nx.from_numpy_array(A)

    m['n_nodes'] = n_nodes
    m['n_edges'] = int(A_b.sum() / 2)
    m['density'] = m['n_edges'] / (n_nodes * (n_nodes - 1) / 2)

    lcc = max(nx.connected_components(G), key=len)
    m['lcc_size'] = len(lcc) / n_nodes
    m['n_components'] = nx.number_connected_components(G)

    eigs = np.linalg.eigvalsh(A_b)
    m['lambda1'] = eigs.max()
    m['tau'] = 1.0 / eigs.max() if eigs.max() > 0 else np.inf

    G_lcc = G.subgraph(lcc).copy()
    if len(G_lcc) > 1:
        m['avg_path'] = nx.average_shortest_path_length(G_lcc)
        m['global_eff'] = nx.global_efficiency(G_lcc)
    else:
        m['avg_path'] = np.inf
        m['global_eff'] = 0.0

    m['avg_clustering'] = nx.average_clustering(G)
    m['mean_strength'] = A.sum(axis=1).mean()
    m['total_strength'] = A.sum()

    return m


def simulate_removal(A, removal_order, metric_keys):
    """Remove nodes in order, collect metrics at each step."""
    records = []
    remaining = list(range(A.shape[0]))
    for step, node in enumerate(removal_order):
        remaining = [r for r in remaining if r != node]
        if len(remaining) < 3:
            break
        sub = A[np.ix_(remaining, remaining)]
        m = network_metrics(sub)
        m['step'] = step + 1
        m['frac_removed'] = (step + 1) / A.shape[0]
        records.append({k: m[k] for k in metric_keys + ['step', 'frac_removed']})
    return pd.DataFrame(records)


def run_removal_simulations(sc_ctx, n, strength, betweenness, bridge_nodes):
    """Run all four removal strategies."""
    KEYS = ['lcc_size', 'global_eff', 'lambda1', 'avg_clustering', 'n_components']

    order_hub = np.argsort(strength)[::-1]
    order_broker = np.argsort(betweenness)[::-1]
    order_bridge = np.concatenate([
        bridge_nodes[np.argsort(betweenness[bridge_nodes])[::-1]],
        np.array([i for i in np.argsort(betweenness)[::-1] if i not in bridge_nodes])
    ])
    order_random = np.random.permutation(n)

    df_hub = simulate_removal(sc_ctx, order_hub, KEYS)
    df_broker = simulate_removal(sc_ctx, order_broker, KEYS)
    df_bridge = simulate_removal(sc_ctx, order_bridge, KEYS)

    rand_dfs = [simulate_removal(sc_ctx, np.random.permutation(n), KEYS) for _ in range(20)]
    df_rand = pd.concat(rand_dfs).groupby('frac_removed').mean().reset_index()

    return df_hub, df_broker, df_bridge, df_rand


def compute_single_node_removal_impact(sc_ctx, sc_ctx_labels, n, baseline,
                                        hub_nodes, broker_nodes, bridge_nodes,
                                        strength, betweenness):
    """Compute metric deltas after removing each key node."""
    key_node_indices = np.concatenate([hub_nodes, broker_nodes, bridge_nodes])
    key_node_labels = [sc_ctx_labels[i] for i in key_node_indices]
    key_node_roles = (['Hub'] * len(hub_nodes) +
                      ['Broker'] * len(broker_nodes) +
                      ['Bridge'] * len(bridge_nodes))

    delta_records = []
    for node, label, role in zip(key_node_indices, key_node_labels, key_node_roles):
        remaining = [i for i in range(n) if i != node]
        sub = sc_ctx[np.ix_(remaining, remaining)]
        m = network_metrics(sub)
        delta_records.append({
            'Region':       label,
            'Role':         role,
            'ΔLCC':         m['lcc_size'] - 1.0,
            'ΔGlobal_Eff':  m['global_eff'] - baseline['global_eff'],
            'Δλ₁':          m['lambda1'] - baseline['lambda1'],
            'ΔClustering':  m['avg_clustering'] - baseline['avg_clustering'],
            'Strength':     strength[node],
            'Betweenness':  betweenness[node],
        })

    return pd.DataFrame(delta_records).sort_values('ΔGlobal_Eff')


def add_node(A, new_weights):
    """Add one new node with given connection weights."""
    n_old = A.shape[0]
    A_new = np.zeros((n_old + 1, n_old + 1))
    A_new[:n_old, :n_old] = A
    A_new[-1, :n_old] = new_weights
    A_new[:n_old, -1] = new_weights
    return A_new


def build_addition_strategies(sc_ctx, n, hub_nodes, strength, degree_bin, G_uw):
    """Build four node-addition strategies and compute metrics."""
    communities = list(nx.community.greedy_modularity_communities(G_uw))
    community_map = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            community_map[node] = cid

    mean_w = sc_ctx[sc_ctx > 0].mean()
    strong_w = np.percentile(sc_ctx[sc_ctx > 0], 75)

    w_hub = np.zeros(n)
    w_hub[hub_nodes] = strong_w
    A_add_hub = add_node(sc_ctx, w_hub)

    w_bridge = np.zeros(n)
    for comm in communities:
        rep = sorted(comm, key=lambda x: strength[x], reverse=True)[0]
        w_bridge[rep] = mean_w
    A_add_bridge = add_node(sc_ctx, w_bridge)

    k_rand = max(len(hub_nodes), 4)
    rand_targets = np.random.choice(n, k_rand, replace=False)
    w_rand = np.zeros(n)
    w_rand[rand_targets] = mean_w
    A_add_rand = add_node(sc_ctx, w_rand)

    low_deg = np.argsort(degree_bin)[:k_rand]
    w_periph = np.zeros(n)
    w_periph[low_deg] = mean_w
    A_add_periph = add_node(sc_ctx, w_periph)

    return [
        ('Hub-like', A_add_hub, w_hub),
        ('Bridge-type', A_add_bridge, w_bridge),
        ('Random', A_add_rand, w_rand),
        ('Peripheral', A_add_periph, w_periph),
    ]


def cascade_failure(A, seed_node, k_min=2):
    """Cascade: remove seed, iteratively remove nodes below k_min degree."""
    A_curr = A.copy()
    n_curr = A_curr.shape[0]
    active = list(range(n_curr))
    removed = [seed_node]
    active.remove(seed_node)
    A_curr[seed_node, :] = 0
    A_curr[:, seed_node] = 0

    changed = True
    while changed:
        changed = False
        degrees = (A_curr[np.ix_(active, active)] > 0).sum(axis=1)
        to_remove = [active[i] for i, d in enumerate(degrees) if d < k_min]
        for node in to_remove:
            if node in active:
                active.remove(node)
                removed.append(node)
                A_curr[node, :] = 0
                A_curr[:, node] = 0
                changed = True

    sub = A_curr[np.ix_(active, active)]
    m = network_metrics(sub) if len(active) > 2 else {'lcc_size': 0, 'global_eff': 0, 'lambda1': 0}
    return removed, active, m
