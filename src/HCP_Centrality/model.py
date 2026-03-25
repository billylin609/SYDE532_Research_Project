# %% model.py — HCP_Centrality: data loading and centrality computation

import numpy as np
import pandas as pd
import networkx as nx

from enigmatoolbox.datasets import load_sc


def load_sc_graph():
    """Load HCP SC and build weighted undirected NetworkX graph."""
    sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()
    n_ctx = sc_ctx.shape[0]

    G = nx.from_numpy_array(sc_ctx)
    label_map = {i: sc_ctx_labels[i] for i in range(n_ctx)}
    nx.set_node_attributes(G, label_map, 'region')

    return sc_ctx, sc_ctx_labels, n_ctx, G


def compute_degree_centrality(G, n_ctx):
    degree_c = nx.degree_centrality(G)
    return np.array([degree_c[i] for i in range(n_ctx)])


def compute_betweenness_centrality(G, n_ctx):
    betweenness_c = nx.betweenness_centrality(G, weight='weight', normalized=True)
    return np.array([betweenness_c[i] for i in range(n_ctx)])


def compute_closeness_centrality(G, n_ctx):
    G_close = G.copy()
    for u, v, d in G_close.edges(data=True):
        d['distance'] = 1.0 / (d['weight'] + 1e-9)
    closeness_c = nx.closeness_centrality(G_close, distance='distance')
    return np.array([closeness_c[i] for i in range(n_ctx)])


def compute_eigenvector_centrality(G, n_ctx):
    eigenvec_c = nx.eigenvector_centrality_numpy(G, weight='weight')
    return np.array([eigenvec_c[i] for i in range(n_ctx)])


def compute_eigenvalue_spectrum(sc_ctx):
    """Return sorted eigenvalues and eigenvectors of the SC matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(sc_ctx)
    eigenvalues_sorted = eigenvalues[::-1]
    eigenvectors_sorted = eigenvectors[:, ::-1]
    lambda1 = eigenvalues_sorted[0]
    v1 = eigenvectors_sorted[:, 0]
    return eigenvalues_sorted, eigenvectors_sorted, lambda1, v1


def build_centrality_df(sc_ctx_labels, degree_arr, betweenness_arr, closeness_arr,
                         eigenvec_arr, v1, lambda1):
    """Build summary DataFrame with centrality measures and ranks."""
    centrality_df = pd.DataFrame({
        'Region':       sc_ctx_labels,
        'Degree':       degree_arr.round(4),
        'Betweenness':  betweenness_arr.round(4),
        'Closeness':    closeness_arr.round(4),
        'Eigenvector':  eigenvec_arr.round(4),
        'v1_component': np.abs(v1).round(4),
    })
    for col in ['Degree', 'Betweenness', 'Closeness', 'Eigenvector', 'v1_component']:
        centrality_df[col + '_rank'] = centrality_df[col].rank(ascending=False).astype(int)
    return centrality_df


def run_all(print_summaries=True):
    """Run full centrality pipeline and return all results."""
    sc_ctx, sc_ctx_labels, n_ctx, G = load_sc_graph()

    degree_arr = compute_degree_centrality(G, n_ctx)
    betweenness_arr = compute_betweenness_centrality(G, n_ctx)
    closeness_arr = compute_closeness_centrality(G, n_ctx)
    eigenvec_arr = compute_eigenvector_centrality(G, n_ctx)
    eigenvalues_sorted, eigenvectors_sorted, lambda1, v1 = compute_eigenvalue_spectrum(sc_ctx)
    centrality_df = build_centrality_df(
        sc_ctx_labels, degree_arr, betweenness_arr, closeness_arr, eigenvec_arr, v1, lambda1
    )

    if print_summaries:
        print(f'Spectral radius λ₁ = {lambda1:.4f}  (global network metric, not per-node)')
        print()
        print(centrality_df[['Region', 'Degree', 'Betweenness', 'Closeness', 'Eigenvector', 'v1_component']]
              .sort_values('Eigenvector', ascending=False)
              .head(15)
              .to_string(index=False))

    return (sc_ctx, sc_ctx_labels, n_ctx, G,
            degree_arr, betweenness_arr, closeness_arr, eigenvec_arr,
            eigenvalues_sorted, lambda1, v1, centrality_df)
