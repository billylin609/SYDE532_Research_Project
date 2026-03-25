# %% model.py — HCP_NeuralNetwork: data loading, graph metrics, efficiency-resilience

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

from enigmatoolbox.datasets import load_sc


def load_sc_graphs():
    """Load HCP SC and build weighted + unweighted graphs."""
    sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()
    n = sc_ctx.shape[0]
    A_bin = (sc_ctx > 0).astype(float)
    G_w = nx.from_numpy_array(sc_ctx)
    G_uw = nx.from_numpy_array(A_bin)
    return sc_ctx, sc_ctx_labels, n, A_bin, G_w, G_uw


def compute_weighted_vs_unweighted(sc_ctx, sc_ctx_labels, G_w, G_uw, n):
    """Compute centrality for weighted and unweighted graphs and correlations."""
    deg_w = np.array([d for _, d in G_w.degree(weight='weight')])
    deg_uw = np.array([d for _, d in G_uw.degree()])

    G_w_dist = G_w.copy()
    for u, v, d in G_w_dist.edges(data=True):
        d['distance'] = 1.0 / (d['weight'] + 1e-9)

    btw_w = np.array(list(nx.betweenness_centrality(G_w_dist, weight='distance', normalized=True).values()))
    btw_uw = np.array(list(nx.betweenness_centrality(G_uw, normalized=True).values()))

    eig_w = np.array(list(nx.eigenvector_centrality_numpy(G_w, weight='weight').values()))
    eig_uw = np.array(list(nx.eigenvector_centrality_numpy(G_uw, weight=None).values()))

    r_deg, _ = stats.spearmanr(deg_w, deg_uw)
    r_btw, _ = stats.spearmanr(btw_w, btw_uw)
    r_eig, _ = stats.spearmanr(eig_w, eig_uw)

    return deg_w, deg_uw, btw_w, btw_uw, eig_w, eig_uw, r_deg, r_btw, r_eig


def detect_communities(G_uw, n, sc_ctx_labels):
    """Greedy modularity community detection."""
    communities = list(nx.community.greedy_modularity_communities(G_uw))
    community_map = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            community_map[node] = comm_id
    community_arr = np.array([community_map[i] for i in range(n)])
    return communities, community_arr


def classify_node_roles(deg_w, btw_w, eig_w, deg_uw, sc_ctx_labels, community_arr, n):
    """Classify nodes as Hub, Broker, Bridge, or Peripheral."""
    strength = deg_w
    between = btw_w
    eigen = eig_w
    degree_bin = deg_uw

    z_str = stats.zscore(strength)
    z_btw = stats.zscore(between)
    z_eig = stats.zscore(eigen)
    z_deg = stats.zscore(degree_bin)

    is_hub = (z_str > 1.0) & (z_eig > 1.0)
    is_broker = (z_btw > 1.0) & ~is_hub
    is_bridge = (z_btw > 0.5) & (z_deg < 0.0) & ~is_hub & ~is_broker

    role_arr = np.full(n, 'Peripheral', dtype=object)
    role_arr[is_bridge] = 'Bridge'
    role_arr[is_broker] = 'Broker'
    role_arr[is_hub] = 'Hub'

    role_df = pd.DataFrame({
        'Region':      sc_ctx_labels,
        'Community':   community_arr,
        'Strength':    strength.round(3),
        'Betweenness': between.round(4),
        'Eigenvector': eigen.round(4),
        'Role':        role_arr,
    })
    return role_df


def compute_topological_metrics(G_uw, A_bin, deg_uw, n):
    """Compute path length, spectral radius, epidemic threshold, node-removal resilience."""
    d_hat = nx.average_shortest_path_length(G_uw)
    eigvals_uw = np.linalg.eigvalsh(A_bin)
    lambda1_uw = eigvals_uw.max()
    tau = 1.0 / lambda1_uw

    hub_node = int(np.argmax(deg_uw))
    A_removed = np.delete(np.delete(A_bin, hub_node, axis=0), hub_node, axis=1)
    lambda1_prime = np.linalg.eigvalsh(A_removed).max()
    Lambda = (lambda1_uw - lambda1_prime) / lambda1_uw * 100

    alpha = Lambda / (d_hat + Lambda)
    beta = (1 / tau) / ((1 / tau) + d_hat)

    return d_hat, lambda1_uw, tau, hub_node, lambda1_prime, Lambda, alpha, beta


def compute_weighted_resilience(sc_ctx, G_uw, n):
    """Compute weighted resilience and efficiency metrics."""
    W_out = sc_ctx.sum(axis=1)
    R_gamma = (W_out.sum() - W_out.max()) / W_out.sum()

    path_lengths = dict(nx.all_pairs_shortest_path_length(G_uw))
    E_vals = []
    for i in range(n):
        for j in range(n):
            if i != j and sc_ctx[i, j] > 0:
                d_ij = path_lengths[i].get(j, np.nan)
                w_ij = sc_ctx[i, j]
                if not np.isnan(d_ij):
                    E_vals.append(d_ij / w_ij)
    E_gamma = np.mean(E_vals)

    E_max = np.max(E_vals)
    E_min = np.min(E_vals)
    E_norm = 1 - (E_gamma - E_min) / (E_max - E_min + 1e-9)
    R_norm = R_gamma

    rho = (1 - E_norm) / (R_norm + 1e-9)
    xi = (1 - E_norm) + R_norm

    return W_out, R_gamma, E_gamma, E_norm, R_norm, rho, xi


def robustness_curve(A, removal_order):
    """Simulate sequential node removal and track LCC, avg path, spectral radius."""
    lcc_sizes, avg_paths, lambdas = [], [], []
    remaining = list(range(A.shape[0]))
    for step, node in enumerate(removal_order):
        remaining = [r for r in remaining if r != node]
        if len(remaining) < 2:
            break
        sub = A[np.ix_(remaining, remaining)]
        G_sub = nx.from_numpy_array((sub > 0).astype(float))
        lcc = max(nx.connected_components(G_sub), key=len)
        lcc_sizes.append(len(lcc) / A.shape[0])
        ev = np.linalg.eigvalsh(sub).max()
        lambdas.append(ev)
        if nx.is_connected(G_sub) and len(remaining) < 30:
            avg_paths.append(nx.average_shortest_path_length(G_sub))
        else:
            avg_paths.append(np.nan)
    return np.array(lcc_sizes), np.array(avg_paths), np.array(lambdas)
