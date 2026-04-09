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


def compute_global_centrality(G, degree_arr, betweenness_arr, closeness_arr, eigenvec_arr, lambda1):
    """
    Network-level (global) summary metrics.
      - Global degree centrality     : mean local degree centrality
      - Global betweenness centrality: mean local betweenness centrality
      - Global closeness centrality  : mean local closeness centrality
      - Graph density                : fraction of possible edges present
      - Spectral radius (λ₁)         : largest eigenvalue of SC matrix
      - Avg weighted clustering coeff: mean local clustering (weights as edge prob)
    """
    return {
        'Global degree centrality':      float(degree_arr.mean()),
        'Global betweenness centrality': float(betweenness_arr.mean()),
        'Global closeness centrality':   float(closeness_arr.mean()),
        'Graph density':                 float(nx.density(G)),
        'Spectral radius (λ₁)':          float(lambda1),
        'Avg clustering coefficient':    float(nx.average_clustering(G, weight='weight')),
    }


def compute_broker_score(degree_arr, betweenness_arr):
    """
    Broker score = normalized betweenness - normalized degree.
    Brokers have high betweenness (sit on many shortest paths) but
    low degree (few direct connections) — bottleneck bridges, not hubs.
    """
    def _norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else np.zeros_like(x)

    return _norm(betweenness_arr) - _norm(degree_arr)


def build_centrality_df(sc_ctx_labels, degree_arr, betweenness_arr, closeness_arr,
                         eigenvec_arr):
    """Build summary DataFrame with centrality measures and ranks."""
    broker_arr = compute_broker_score(degree_arr, betweenness_arr)

    centrality_df = pd.DataFrame({
        'Region':       sc_ctx_labels,
        'Degree':       degree_arr.round(4),
        'Betweenness':  betweenness_arr.round(4),
        'Closeness':    closeness_arr.round(4),
        'Eigenvector':  eigenvec_arr.round(4),
        'Broker_score': broker_arr.round(4),
    })
    for col in ['Degree', 'Betweenness', 'Closeness', 'Eigenvector', 'Broker_score']:
        centrality_df[col + '_rank'] = centrality_df[col].rank(ascending=False).astype(int)

    threshold = broker_arr.mean() + broker_arr.std()
    centrality_df['is_broker'] = broker_arr > threshold

    return centrality_df, broker_arr


def run_all(print_summaries=True):
    """Run full centrality pipeline and return all results."""
    sc_ctx, sc_ctx_labels, n_ctx, G = load_sc_graph()

    degree_arr     = compute_degree_centrality(G, n_ctx)
    betweenness_arr = compute_betweenness_centrality(G, n_ctx)
    closeness_arr  = compute_closeness_centrality(G, n_ctx)
    eigenvec_arr   = compute_eigenvector_centrality(G, n_ctx)
    eigenvalues_sorted, eigenvectors_sorted, lambda1, v1 = compute_eigenvalue_spectrum(sc_ctx)
    centrality_df, broker_arr = build_centrality_df(
        sc_ctx_labels, degree_arr, betweenness_arr, closeness_arr, eigenvec_arr
    )
    global_metrics = compute_global_centrality(
        G, degree_arr, betweenness_arr, closeness_arr, eigenvec_arr, lambda1
    )

    if print_summaries:
        # 1. Local centrality — top 15 by Eigenvector
        print('=' * 60)
        print('LOCAL CENTRALITY  (top 15 by Eigenvector)')
        print('=' * 60)
        print(centrality_df[['Region', 'Degree', 'Betweenness', 'Closeness', 'Eigenvector']]
              .sort_values('Eigenvector', ascending=False)
              .head(15)
              .to_string(index=False))

        # 2. Broker nodes
        print()
        print('=' * 60)
        brokers = centrality_df[centrality_df['is_broker']].sort_values('Broker_score', ascending=False)
        print(f'BROKER NODES  ({len(brokers)} identified — high betweenness, low degree)')
        print('=' * 60)
        print(brokers[['Region', 'Degree_rank', 'Betweenness_rank', 'Broker_score']]
              .to_string(index=False))

        # 3. Global centrality
        print()
        print('=' * 60)
        print('GLOBAL CENTRALITY')
        print('=' * 60)
        for k, v in global_metrics.items():
            print(f'  {k:<38} {v:.6f}')

    return (sc_ctx, sc_ctx_labels, n_ctx, G,
            degree_arr, betweenness_arr, closeness_arr, eigenvec_arr,
            broker_arr, global_metrics, eigenvalues_sorted, lambda1, v1, centrality_df)
