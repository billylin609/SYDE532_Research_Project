# %% model.py — HCP_Community: data loading and community detection

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import igraph as ig
import leidenalg
import infomap

from enigmatoolbox.datasets import load_sc

np.random.seed(42)

# ── Functional atlas definition ────────────────────────────────────────────────
FUNCTIONAL_ATLAS = {
    'Default Mode Network (DMN)': [
        'posteriorcingulate', 'isthmuscingulate', 'precuneus',
        'medialorbitofrontal', 'frontalpole', 'entorhinal',
        'parahippocampal', 'middletemporal', 'inferiortemporal',
        'angularg', 'bankssts',
    ],
    'Sensorimotor': [
        'precentral', 'postcentral', 'paracentral',
        'superiorfrontal',
        'rostralmiddlefrontal',
    ],
    'Visual': [
        'cuneus', 'lingual', 'pericalcarine',
        'lateraloccipital', 'fusiform', 'lingual',
    ],
    'Frontoparietal (Executive)': [
        'caudalmiddlefrontal', 'parsopercularis', 'parsorbitalis',
        'parstriangularis', 'lateralorbitofrontal',
        'superiorparietal', 'inferiorparietal', 'supramarginal',
    ],
    'Temporal / Language': [
        'superiortemporal', 'transversetemporal',
        'temporalpole', 'caudalanteriorcingulate',
    ],
    'Limbic': [
        'entorhinal', 'parahippocampal', 'temporalpole',
        'caudalanteriorcingulate', 'rostralanteriorcingulate',
    ],
    'Salience / Cingulo-Opercular': [
        'insula', 'rostralanteriorcingulate', 'caudalanteriorcingulate',
    ],
}

NETWORK_COLORS = {
    'Default Mode Network (DMN)': '#E05C5C',
    'Sensorimotor':               '#4A90D9',
    'Visual':                     '#8E44AD',
    'Frontoparietal (Executive)': '#27AE60',
    'Temporal / Language':        '#F39C12',
    'Limbic':                     '#1ABC9C',
    'Salience / Cingulo-Opercular': '#E67E22',
    'Unassigned':                 '#BDC3C7',
}


def assign_known_network(region_short):
    """Return best-matching functional network for a DK region."""
    r = region_short.lower()
    matches = {net: sum(k in r for k in keywords)
               for net, keywords in FUNCTIONAL_ATLAS.items()}
    best_net, best_score = max(matches.items(), key=lambda x: x[1])
    return best_net if best_score > 0 else 'Unassigned'


def load_full_sc():
    """Load full 82x82 SC (68 cortical + 14 subcortical) and assign network labels."""
    sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()
    n_ctx  = sc_ctx.shape[0]   # 68
    n_sctx = sc_sctx.shape[0]  # 14
    n      = n_ctx + n_sctx    # 82

    # Build symmetric 82x82 matrix
    A_full = np.zeros((n, n))
    A_full[:n_ctx, :n_ctx] = sc_ctx           # cortical-cortical
    A_full[n_ctx:, :n_ctx] = sc_sctx          # subcortical-cortical
    A_full[:n_ctx, n_ctx:] = sc_sctx.T        # cortical-subcortical

    all_labels = np.concatenate([sc_ctx_labels, sc_sctx_labels])
    A_bin = (A_full > 0).astype(float)
    G_w   = nx.from_numpy_array(A_full)
    G_uw  = nx.from_numpy_array(A_bin)

    short_labels    = [l.replace('L_', '').replace('R_', '').lower() for l in all_labels]
    known_networks  = [assign_known_network(s) for s in short_labels]

    return A_full, all_labels, n, A_bin, G_w, G_uw, short_labels, known_networks


def load_sc_with_networks():
    """Load HCP SC and assign functional network labels."""
    sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()
    n = sc_ctx.shape[0]
    A_bin = (sc_ctx > 0).astype(float)
    G_w = nx.from_numpy_array(sc_ctx)
    G_uw = nx.from_numpy_array(A_bin)

    short_labels = [l.replace('L_', '').replace('R_', '').lower() for l in sc_ctx_labels]
    known_networks = [assign_known_network(s) for s in short_labels]

    return sc_ctx, sc_ctx_labels, n, A_bin, G_w, G_uw, short_labels, known_networks


def detect_greedy_modularity(G_uw):
    """Greedy modularity community detection."""
    communities_gm = list(nx.community.greedy_modularity_communities(G_uw))
    labels_gm = np.zeros(len(list(G_uw.nodes())), dtype=int)
    for cid, comm in enumerate(communities_gm):
        for node in comm:
            labels_gm[node] = cid
    Q_gm = nx.community.modularity(G_uw, communities_gm)
    return labels_gm, Q_gm, communities_gm


def detect_leiden(sc_ctx, n):
    """Leiden algorithm community detection."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if sc_ctx[i, j] > 0]
    weights = [sc_ctx[i, j] for i, j in edges]
    G_ig = ig.Graph(n=n, edges=edges)
    G_ig.es['weight'] = weights
    partition_leiden = leidenalg.find_partition(
        G_ig, leidenalg.ModularityVertexPartition, weights='weight', seed=42
    )
    labels_leiden = np.array(partition_leiden.membership)
    Q_leiden = partition_leiden.modularity
    return labels_leiden, Q_leiden


def detect_infomap(G_w, G_uw, n):
    """Infomap community detection."""
    im = infomap.Infomap('--two-level --seed 42 --silent')
    for i, j in G_w.edges():
        w = G_w[i][j]['weight']
        im.add_link(i, j, w)
    im.run()

    labels_im = np.zeros(n, dtype=int)
    for node in im.nodes:
        labels_im[node.node_id] = node.module_id - 1

    n_comm_im = len(set(labels_im))
    communities_im = [set(np.where(labels_im == c)[0]) for c in range(n_comm_im)]
    Q_im = nx.community.modularity(G_uw, communities_im)
    return labels_im, Q_im, n_comm_im


def detect_label_propagation(G_uw, n):
    """Label propagation (10-run consensus)."""
    all_lp = []
    for seed in range(10):
        np.random.seed(seed)
        comms_lp = list(nx.community.label_propagation_communities(G_uw))
        lab = np.zeros(n, dtype=int)
        for cid, comm in enumerate(comms_lp):
            for node in comm:
                lab[node] = cid
        all_lp.append(lab)

    labels_lp = stats.mode(np.vstack(all_lp), axis=0).mode.flatten()
    _, labels_lp = np.unique(labels_lp, return_inverse=True)
    communities_lp = [set(np.where(labels_lp == c)[0]) for c in np.unique(labels_lp)]
    Q_lp = nx.community.modularity(G_uw, communities_lp)
    return labels_lp, Q_lp


def detect_spectral(sc_ctx, k_spectral, G_uw):
    """Spectral clustering."""
    sc_model = SpectralClustering(
        n_clusters=k_spectral, affinity='precomputed',
        random_state=42, n_init=20
    )
    labels_spectral = sc_model.fit_predict(sc_ctx)
    communities_spectral = [set(np.where(labels_spectral == c)[0]) for c in range(k_spectral)]
    Q_spectral = nx.community.modularity(G_uw, communities_spectral)
    return labels_spectral, Q_spectral


def run_all_algorithms(sc_ctx, n, G_w, G_uw):
    """Run all 5 community detection algorithms."""
    labels_gm, Q_gm, _ = detect_greedy_modularity(G_uw)
    labels_leiden, Q_leiden = detect_leiden(sc_ctx, n)
    labels_im, Q_im, _ = detect_infomap(G_w, G_uw, n)
    labels_lp, Q_lp = detect_label_propagation(G_uw, n)
    k_spectral = len(set(labels_leiden))
    labels_spectral, Q_spectral = detect_spectral(sc_ctx, k_spectral, G_uw)

    all_labels = {
        'Greedy Modularity': labels_gm,
        'Leiden':            labels_leiden,
        'Infomap':           labels_im,
        'Label Propagation': labels_lp,
        'Spectral':          labels_spectral,
    }
    all_Q = {
        'Greedy Modularity': Q_gm,
        'Leiden':            Q_leiden,
        'Infomap':           Q_im,
        'Label Propagation': Q_lp,
        'Spectral':          Q_spectral,
    }
    return all_labels, all_Q


def build_summary_df(all_labels, all_Q):
    """Build summary table of algorithm comparison."""
    summary_rows = []
    for name, labs in all_labels.items():
        sizes = [int((labs == c).sum()) for c in np.unique(labs)]
        summary_rows.append({
            'Algorithm':     name,
            'N Communities': len(np.unique(labs)),
            'Modularity Q':  round(all_Q[name], 4),
            'Min size':      min(sizes),
            'Max size':      max(sizes),
            'Mean size':     round(np.mean(sizes), 1),
        })
    return pd.DataFrame(summary_rows)


def compute_nmi_matrix(all_labels):
    """NMI between all algorithm pairs."""
    alg_names = list(all_labels.keys())
    nmi_matrix = np.zeros((len(alg_names), len(alg_names)))
    for i, n1 in enumerate(alg_names):
        for j, n2 in enumerate(alg_names):
            nmi_matrix[i, j] = normalized_mutual_info_score(all_labels[n1], all_labels[n2])
    return nmi_matrix, alg_names


def label_communities(labels_arr, method_name, sc_ctx_labels, known_networks,
                       strength_arr, btw_arr):
    """Assign functional label to each community by majority vote."""
    records = []
    for cid in np.unique(labels_arr):
        members_idx = np.where(labels_arr == cid)[0]
        members_name = sc_ctx_labels[members_idx]
        net_counts = pd.Series([known_networks[i] for i in members_idx]).value_counts()
        top_network = net_counts.index[0]
        purity = net_counts.iloc[0] / len(members_idx)
        records.append({
            'Method':           method_name,
            'Community ID':     cid,
            'Size':             len(members_idx),
            'Functional Label': top_network,
            'Purity':           round(purity, 3),
            'Avg Strength':     round(strength_arr[members_idx].mean(), 3),
            'Avg Betweenness':  round(btw_arr[members_idx].mean(), 5),
            'Members':          list(members_name),
        })
    return pd.DataFrame(records)


def compute_global_network_metrics(sc_ctx, G_w, G_uw):
    """
    Network-level (global) structural metrics.
      - Nodes / Edges
      - Total streamline weight
      - Graph density
      - Average weighted degree (strength)
      - Average weighted clustering coefficient
      - Transitivity (global clustering coefficient)
      - Degree assortativity
    """
    m = sc_ctx.sum() / 2
    return {
        'Nodes':                         G_uw.number_of_nodes(),
        'Edges':                         G_uw.number_of_edges(),
        'Total streamline weight':        float(m),
        'Graph density':                  float(nx.density(G_uw)),
        'Avg strength (weighted degree)': float(sc_ctx.sum(axis=1).mean()),
        'Avg weighted clustering coeff':  float(nx.average_clustering(G_w, weight='weight')),
        'Transitivity':                   float(nx.transitivity(G_uw)),
        'Degree assortativity':           float(nx.degree_assortativity_coefficient(G_uw)),
    }


def compute_per_community_Q(sc_ctx, labels):
    """
    Per-community modularity contribution.
    Q_c = e_c - a_c^2
      e_c = fraction of total weight inside community c
      a_c = fraction of total strength belonging to community c
    Sum over all communities equals the global Q.
    """
    two_m    = sc_ctx.sum()          # 2m (sum of all weights)
    strength = sc_ctx.sum(axis=1)    # node strengths

    records = []
    for cid in np.unique(labels):
        members = np.where(labels == cid)[0]
        e_c = sc_ctx[np.ix_(members, members)].sum() / two_m
        a_c = strength[members].sum() / two_m
        Q_c = e_c - a_c ** 2
        records.append({
            'Community':  cid,
            'Size':       len(members),
            'e_c (within weight frac)': round(float(e_c), 5),
            'a_c (strength frac)':      round(float(a_c), 5),
            'Q_c (contribution)':       round(float(Q_c), 5),
        })
    return pd.DataFrame(records)


def participation_coefficient(A, labels):
    """Participation coefficient per node."""
    n_nodes = A.shape[0]
    pc = np.zeros(n_nodes)
    for i in range(n_nodes):
        ki = A[i].sum()
        if ki == 0:
            continue
        pc[i] = 1 - sum(
            (A[i][labels == c].sum() / ki) ** 2
            for c in np.unique(labels)
        )
    return pc


def intra_z_score(A, labels):
    """Intra-community z-score per node."""
    z = np.zeros(A.shape[0])
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        k_intra = A[np.ix_(members, members)].sum(axis=1)
        if k_intra.std() > 0:
            z[members] = (k_intra - k_intra.mean()) / k_intra.std()
    return z


def cartographic_role(z, pc):
    """Classify into hub/non-hub connector/peripheral roles (Guimerà & Amaral 2005)."""
    roles = []
    for zi, pci in zip(z, pc):
        if zi >= 2.5:
            if pci < 0.30:   roles.append('Provincial Hub')
            elif pci < 0.75: roles.append('Connector Hub')
            else:             roles.append('Kinless Hub')
        else:
            if pci < 0.05:   roles.append('Ultra-peripheral')
            elif pci < 0.62: roles.append('Peripheral')
            elif pci < 0.80: roles.append('Connector')
            else:             roles.append('Kinless')
    return roles


def compute_between_community_strength(sc_ctx, labels_leiden):
    """Compute within- and between-community connectivity matrices."""
    unique_comms = np.unique(labels_leiden)
    n_comm = len(unique_comms)

    within_density = np.zeros(n_comm)
    within_strength = np.zeros(n_comm)
    between_strength = np.zeros((n_comm, n_comm))

    for i, ci in enumerate(unique_comms):
        mi = np.where(labels_leiden == ci)[0]
        sub_w = sc_ctx[np.ix_(mi, mi)]
        sub_b = (sub_w > 0).astype(float)
        n_i = len(mi)
        possible = n_i * (n_i - 1) / 2
        within_density[i] = sub_b.sum() / 2 / possible if possible > 0 else 0
        within_strength[i] = sub_w.sum() / 2

        for j, cj in enumerate(unique_comms):
            if j > i:
                mj = np.where(labels_leiden == cj)[0]
                bw = sc_ctx[np.ix_(mi, mj)].sum()
                between_strength[i, j] = bw
                between_strength[j, i] = bw

    return within_density, within_strength, between_strength, unique_comms, n_comm
