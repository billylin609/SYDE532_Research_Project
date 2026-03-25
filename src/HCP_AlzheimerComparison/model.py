# %% model.py — HCP_AlzheimerComparison: data loading and AD cascade simulations

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

from enigmatoolbox.datasets import load_sc, load_fc

np.random.seed(42)

BRAAK_KEYWORDS = {
    'I-II (Transentorhinal)':  ['entorhinal', 'parahippocampal'],
    'III-IV (Limbic)':         ['isthmuscingulate', 'inferiortemporal',
                                 'middletemporal', 'fusiform', 'lingual'],
    'V-VI (Isocortical)':      ['posteriorcingulate', 'precuneus',
                                 'inferiorparietal', 'superiorparietal',
                                 'lateraloccipital', 'superiorfrontal',
                                 'rostralmiddlefrontal', 'caudalmiddlefrontal'],
}

DMN_KEYWORDS = ['posteriorcingulate', 'precuneus', 'inferiorparietal',
                 'middletemporal', 'superiorfrontal', 'medialorbitofrontal']


def match_regions(keywords, labels):
    """Return list of (index, label) pairs where label contains any keyword."""
    out = []
    for i, lab in enumerate(labels):
        lab_low = lab.lower()
        if any(kw in lab_low for kw in keywords):
            out.append((i, lab))
    return out


def load_sc_and_classify():
    """Load HCP SC, classify Braak regions, compute centrality."""
    sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()
    N = sc_ctx.shape[0]
    labels = list(sc_ctx_labels)

    braak_indices = {}
    for stage, kws in BRAAK_KEYWORDS.items():
        matched = match_regions(kws, labels)
        braak_indices[stage] = [i for i, _ in matched]

    dmn_indices = [i for i, _ in match_regions(DMN_KEYWORDS, labels)]

    G_full = nx.from_numpy_array(sc_ctx)
    bet0 = nx.betweenness_centrality(G_full, normalized=False)
    deg0 = dict(G_full.degree(weight='weight'))
    eig0 = nx.eigenvector_centrality_numpy(G_full, weight='weight')

    bet_arr = np.array([bet0[i] for i in range(N)])
    deg_arr = np.array([deg0[i] for i in range(N)])
    eig_arr = np.array([eig0[i] for i in range(N)])

    return sc_ctx, labels, N, braak_indices, dmn_indices, bet_arr, deg_arr, eig_arr


def global_efficiency(adj):
    G = nx.from_numpy_array((adj > 0).astype(float))
    return nx.global_efficiency(G)


def network_metrics_ad(adj, removed_set):
    """Compute LCC fraction and global efficiency after removing nodes."""
    active = [i for i in range(adj.shape[0]) if i not in removed_set]
    if len(active) < 2:
        return 0.0, 0.0
    sub = adj[np.ix_(active, active)]
    G = nx.from_numpy_array((sub > 0).astype(float))
    comps = list(nx.connected_components(G))
    lcc = max(len(c) for c in comps) / adj.shape[0]
    eff = nx.global_efficiency(G)
    return lcc, eff


def simulate_braak_removal(sc_ctx, braak_indices):
    """Sequentially remove Braak-staged regions and measure network metrics."""
    braak_stages = list(BRAAK_KEYWORDS.keys())
    braak_results = []
    removed = set()

    lcc_base, eff_base = network_metrics_ad(sc_ctx, removed)
    braak_results.append({'stage': 'Baseline', 'n_removed': 0,
                           'lcc': lcc_base, 'eff': eff_base})

    for stage in braak_stages:
        removed.update(braak_indices[stage])
        lcc, eff = network_metrics_ad(sc_ctx, removed)
        braak_results.append({'stage': stage,
                               'n_removed': len(removed),
                               'lcc': lcc, 'eff': eff})

    return pd.DataFrame(braak_results), eff_base


def motter_lai(A, attack_node, alpha, weighted=False, return_history=False):
    """Motter-Lai cascade (identical implementation to HCP_MotterLai)."""
    N_total = A.shape[0]

    def build_graph(adj, active):
        sub = adj[np.ix_(active, active)]
        if weighted:
            G = nx.from_numpy_array(sub)
            for u, v, d in G.edges(data=True):
                d['distance'] = 1.0 / (d['weight'] + 1e-9)
            return G
        return nx.from_numpy_array((sub > 0).astype(float))

    def get_betweenness(G, wt=False):
        if wt:
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
        Gs = build_graph(A, active)
        Ls = get_betweenness(Gs, weighted)
        overloaded = set()
        for new_idx, orig_idx in enumerate(active):
            if Ls[new_idx] > C[orig_idx]:
                overloaded.add(orig_idx)
        if not overloaded:
            break
        removed.update(overloaded)
        history.append(overloaded)
        active = [i for i in active if i not in overloaded]

    if len(active) == 0:
        G_metric = 0.0
    else:
        sub = A[np.ix_(active, active)]
        Gf = nx.from_numpy_array((sub > 0).astype(float))
        comps = list(nx.connected_components(Gf))
        G_metric = max(len(c) for c in comps) / N_total

    n_failed = len(removed)
    if return_history:
        return G_metric, n_failed, history
    return G_metric, n_failed


def motter_lai_multiseed(A, attack_nodes, alpha):
    """Multi-seed Motter-Lai: remove a set of nodes then cascade."""
    N_total = A.shape[0]

    G0 = nx.from_numpy_array((A > 0).astype(float))
    L0 = nx.betweenness_centrality(G0, normalized=False)
    C = {i: (1 + alpha) * L0[i] for i in range(N_total)}

    active = [i for i in range(N_total) if i not in set(attack_nodes)]
    removed = set(attack_nodes)

    for _ in range(N_total):
        if len(active) < 2:
            break
        sub = A[np.ix_(active, active)]
        Gs = nx.from_numpy_array((sub > 0).astype(float))
        Ls = nx.betweenness_centrality(Gs, normalized=False)
        overloaded = set()
        for new_i, orig_i in enumerate(active):
            if Ls[new_i] > C[orig_i]:
                overloaded.add(orig_i)
        if not overloaded:
            break
        removed.update(overloaded)
        active = [i for i in active if i not in overloaded]

    if len(active) == 0:
        return 0.0, len(removed)
    sub = A[np.ix_(active, active)]
    Gf = nx.from_numpy_array((sub > 0).astype(float))
    comps = list(nx.connected_components(Gf))
    G_metric = max(len(c) for c in comps) / N_total
    return G_metric, len(removed)


def sweep_entorhinal_vs_hub(sc_ctx, N, labels, braak_indices, bet_arr, ALPHAS=None):
    """Sweep alpha for entorhinal and hub attacks."""
    if ALPHAS is None:
        ALPHAS = np.linspace(0.0, 1.0, 20)

    entorhinal_idx = [i for i, lab in enumerate(labels) if 'entorhinal' in lab.lower()]
    top_bet_node = int(np.argmax(bet_arr))

    all_braak_all = set(braak_indices['I-II (Transentorhinal)'] +
                        braak_indices['III-IV (Limbic)'] +
                        braak_indices['V-VI (Isocortical)'])
    non_ad_indices = [i for i in range(N) if i not in all_braak_all]

    results = {'alpha': ALPHAS}
    for key in ['entorhinal_lh', 'entorhinal_rh', 'top_hub', 'random_mean', 'random_std']:
        results[key] = np.zeros(len(ALPHAS))

    for ai, alpha in enumerate(ALPHAS):
        for k, idx in enumerate(entorhinal_idx[:2]):
            key = ['entorhinal_lh', 'entorhinal_rh'][k]
            g, _ = motter_lai(sc_ctx, idx, alpha)
            results[key][ai] = g
        g, _ = motter_lai(sc_ctx, top_bet_node, alpha)
        results['top_hub'][ai] = g
        rng_seeds = np.random.choice(non_ad_indices, size=15, replace=False)
        g_vals = [motter_lai(sc_ctx, s, alpha)[0] for s in rng_seeds]
        results['random_mean'][ai] = np.mean(g_vals)
        results['random_std'][ai] = np.std(g_vals)

    return results, entorhinal_idx, top_bet_node, all_braak_all, non_ad_indices


def sweep_braak_multiseed(sc_ctx, braak_indices, ALPHA_TEST=None):
    """Multi-seed Braak cascade across alpha values."""
    if ALPHA_TEST is None:
        ALPHA_TEST = [0.1, 0.3, 0.5, 1.0]
    STAGE_KEYS = ['I-II (Transentorhinal)', 'III-IV (Limbic)', 'V-VI (Isocortical)']

    multi_results = {a: [] for a in ALPHA_TEST}

    for alpha in ALPHA_TEST:
        cumulative_removed = set()
        multi_results[alpha].append({'stage': 'Baseline', 'G': 1.0, 'total_lost': 0})
        for stage in STAGE_KEYS:
            cumulative_removed.update(braak_indices[stage])
            g, n_lost = motter_lai_multiseed(sc_ctx, list(cumulative_removed), alpha)
            multi_results[alpha].append({'stage': stage, 'G': g, 'total_lost': n_lost})

    return multi_results, ALPHA_TEST, STAGE_KEYS


def compute_vulnerability_map(sc_ctx, N, bet_arr, braak_indices, ALPHA_VUL=0.2):
    """Compute per-node ML vulnerability and biological vulnerability scores."""
    G_per_node = np.zeros(N)
    for i in range(N):
        g, _ = motter_lai(sc_ctx, i, ALPHA_VUL)
        G_per_node[i] = g

    vulnerability_ml = 1 - G_per_node

    bio_score = np.zeros(N)
    for i in range(N):
        if i in braak_indices['I-II (Transentorhinal)']:
            bio_score[i] = 3.0
        elif i in braak_indices['III-IV (Limbic)']:
            bio_score[i] = 2.0
        elif i in braak_indices['V-VI (Isocortical)']:
            bio_score[i] = 1.0

    bet_norm = (bet_arr - bet_arr.min()) / (bet_arr.max() - bet_arr.min() + 1e-9)
    bio_score_continuous = 0.6 * bio_score / 3.0 + 0.4 * bet_norm

    r, p = stats.spearmanr(bio_score_continuous, vulnerability_ml)

    return G_per_node, vulnerability_ml, bio_score, bio_score_continuous, r, p
