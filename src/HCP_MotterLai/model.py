# %% model.py — HCP_MotterLai: data loading and Motter-Lai cascade simulation

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

from enigmatoolbox.datasets import load_sc

np.random.seed(42)


def load_sc_and_loads():
    """Load HCP SC and compute initial loads for Motter-Lai."""
    sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()
    N = sc_ctx.shape[0]
    A_bin = (sc_ctx > 0).astype(float)

    G_w = nx.from_numpy_array(sc_ctx)
    G_uw = nx.from_numpy_array(A_bin)

    G_dist = G_w.copy()
    for u, v, d in G_dist.edges(data=True):
        d['distance'] = 1.0 / (d['weight'] + 1e-9)

    L0_uw = nx.betweenness_centrality(G_uw, normalized=False)
    L0_w = nx.betweenness_centrality(G_dist, weight='distance', normalized=False)

    L0_uw_arr = np.array([L0_uw[i] for i in range(N)])
    L0_w_arr = np.array([L0_w[i] for i in range(N)])

    degree_arr = np.array([d for _, d in G_uw.degree()])

    return sc_ctx, sc_ctx_labels, N, A_bin, G_w, G_uw, L0_uw_arr, L0_w_arr, degree_arr


def motter_lai(A, attack_node, alpha, weighted=False, return_history=False):
    """
    Simulate one Motter-Lai cascade.

    Parameters
    ----------
    A           : (N, N) adjacency matrix
    attack_node : index of initially removed node
    alpha       : tolerance parameter  C_j = (1+alpha) * L_j(0)
    weighted    : if True use inverse-weight shortest paths for betweenness
    return_history : if True also return per-round removal history

    Returns
    -------
    G_metric   : N_prime / N  (fraction surviving in LCC)
    n_failed   : total nodes failed (including seed)
    history    : list of sets of nodes removed per round (if return_history)
    """
    N_total = A.shape[0]

    def build_graph(adj, active):
        sub = adj[np.ix_(active, active)]
        if weighted:
            G = nx.from_numpy_array(sub)
            for u, v, d in G.edges(data=True):
                d['distance'] = 1.0 / (d['weight'] + 1e-9)
            return G
        return nx.from_numpy_array((sub > 0).astype(float))

    def get_betweenness(G, weighted=False):
        if weighted:
            return nx.betweenness_centrality(G, weight='distance', normalized=False)
        return nx.betweenness_centrality(G, normalized=False)

    all_nodes = list(range(N_total))
    G0 = build_graph(A, all_nodes)
    L0 = get_betweenness(G0, weighted)
    C = {i: (1 + alpha) * L0[i] for i in range(N_total)}

    active = [i for i in all_nodes if i != attack_node]
    removed = {attack_node}
    history = [{attack_node}]

    for _ in range(N_total):
        if len(active) < 2:
            break
        G_curr = build_graph(A, active)
        L_curr = get_betweenness(G_curr, weighted)

        overloaded_local = [j for j, node in enumerate(active)
                            if L_curr[j] > C[node]]
        overloaded_global = {active[j] for j in overloaded_local}

        if not overloaded_global:
            break

        active = [i for i in active if i not in overloaded_global]
        removed |= overloaded_global
        history.append(overloaded_global)

    if len(active) >= 1:
        G_final = nx.from_numpy_array((A[np.ix_(active, active)] > 0).astype(float))
        lcc = max(nx.connected_components(G_final), key=len) if G_final.number_of_nodes() > 0 else set()
        G_metric = len(lcc) / N_total
    else:
        G_metric = 0.0

    n_failed = len(removed)

    if return_history:
        return G_metric, n_failed, history
    return G_metric, n_failed


def sweep_alpha(sc_ctx, node_max_load, node_max_degree, N, alphas=None, n_rand=20):
    """Sweep alpha values for load, degree, and random attacks."""
    if alphas is None:
        alphas = np.linspace(0, 1.0, 30)

    G_load, G_degree, G_rand = [], [], []

    for alpha in alphas:
        g, _ = motter_lai(sc_ctx, node_max_load, alpha)
        G_load.append(g)

        g, _ = motter_lai(sc_ctx, node_max_degree, alpha)
        G_degree.append(g)

        rand_nodes = np.random.choice(N, n_rand, replace=False)
        g_rand = np.mean([motter_lai(sc_ctx, int(rn), alpha)[0] for rn in rand_nodes])
        G_rand.append(g_rand)

    return alphas, G_load, G_degree, G_rand


def compute_per_node_vulnerability(sc_ctx, N, alpha_low, alpha_med, alpha_high, load_arr, degree_arr):
    """Compute G at three alpha values for every node."""
    node_results = []
    for i in range(N):
        g_low, nf_low = motter_lai(sc_ctx, i, alpha_low)
        g_med, nf_med = motter_lai(sc_ctx, i, alpha_med)
        g_high, nf_high = motter_lai(sc_ctx, i, alpha_high)
        node_results.append({
            'Node':    i,
            'L0':      load_arr[i],
            'Degree':  degree_arr[i],
            f'G_α={alpha_low}':  g_low,
            f'G_α={alpha_med}':  g_med,
            f'G_α={alpha_high}': g_high,
            f'Failed_α={alpha_low}':  nf_low,
            f'Failed_α={alpha_med}':  nf_med,
            f'Failed_α={alpha_high}': nf_high,
        })
    return pd.DataFrame(node_results)


def classify_failure_mode(g_low, g_med, g_high):
    """Classify node failure mode."""
    if g_low < 0.70:
        return 'Catastrophic'
    elif g_low < 0.85:
        return 'Cascading'
    elif g_low < 0.95:
        return 'Contained'
    else:
        return 'Negligible'


def compute_weighted_vs_unweighted(sc_ctx, N, L0_w_arr, L0_uw_arr):
    """Compare weighted vs unweighted G per node at alpha=0.2."""
    node_max_load_w = int(np.argmax(L0_w_arr))
    node_max_load_uw = int(np.argmax(L0_uw_arr))

    alphas_short = np.linspace(0, 0.8, 20)
    G_uw_load = [motter_lai(sc_ctx, node_max_load_uw, a, weighted=False)[0] for a in alphas_short]
    G_w_load = [motter_lai(sc_ctx, node_max_load_w, a, weighted=True)[0] for a in alphas_short]

    G_per_node_uw = [motter_lai(sc_ctx, i, 0.2, weighted=False)[0] for i in range(N)]
    G_per_node_w = [motter_lai(sc_ctx, i, 0.2, weighted=True)[0] for i in range(N)]

    r, _ = stats.pearsonr(G_per_node_uw, G_per_node_w)
    return (node_max_load_w, node_max_load_uw, alphas_short,
            G_uw_load, G_w_load, G_per_node_uw, G_per_node_w, r)
