# %% model.py — HCP_Visualize: data loading and computation

import numpy as np
import pandas as pd
from scipy import stats

from enigmatoolbox.datasets import load_sc, load_fc


def load_sc_data():
    """Load and return SC matrices plus combined block matrix."""
    sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()
    n_ctx = sc_ctx.shape[0]
    n_sctx = sc_sctx.shape[0]

    sc_combined = np.zeros((n_ctx + n_sctx, n_ctx + n_sctx))
    sc_combined[:n_ctx, :n_ctx] = sc_ctx
    sc_combined[n_ctx:, :n_ctx] = sc_sctx
    sc_combined[:n_ctx, n_ctx:] = sc_sctx.T
    combined_labels = list(sc_ctx_labels) + list(sc_sctx_labels)

    return sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels, sc_combined, combined_labels


def load_fc_data():
    """Load and return FC matrices."""
    return load_fc()


def compute_weight_stats(sc_ctx):
    """Return density and nonzero weights for the cortical SC matrix."""
    n = sc_ctx.shape[0]
    total_possible = n * (n - 1) / 2
    upper = sc_ctx[np.triu_indices(n, k=1)]
    nonzero_weights = upper[upper > 0]
    density = len(nonzero_weights) / total_possible
    return density, nonzero_weights


def compute_hub_df(sc_ctx, sc_ctx_labels):
    """Return DataFrame of regions sorted by weighted strength."""
    sc_strength = sc_ctx.sum(axis=1)
    hub_df = pd.DataFrame({'Region': sc_ctx_labels, 'Strength': sc_strength})
    hub_df = hub_df.sort_values('Strength', ascending=False).reset_index(drop=True)
    sc_bin_degree = (sc_ctx > 0).sum(axis=1)
    return hub_df, sc_strength, sc_bin_degree


def compute_hemisphere_symmetry(sc_ctx, sc_ctx_labels, sc_strength):
    """Compute LH/RH split and asymmetry index."""
    n_ctx = sc_ctx.shape[0]
    n_hemi = n_ctx // 2
    lh_strength = sc_strength[:n_hemi]
    rh_strength = sc_strength[n_hemi:]
    lh_labels = [l.replace('lh_', '') for l in sc_ctx_labels[:n_hemi]]
    corr_lr = np.corrcoef(lh_strength, rh_strength)[0, 1]
    asymmetry = (lh_strength - rh_strength) / (lh_strength + rh_strength + 1e-9)
    return lh_strength, rh_strength, lh_labels, corr_lr, asymmetry


def rich_club_coefficient(W, k_levels=None):
    """Weighted rich-club coefficient across degree thresholds."""
    deg = (W > 0).sum(axis=1)
    if k_levels is None:
        k_levels = np.arange(1, deg.max())
    rc = []
    for k in k_levels:
        nodes_k = np.where(deg > k)[0]
        if len(nodes_k) < 2:
            rc.append(np.nan)
            continue
        sub = W[np.ix_(nodes_k, nodes_k)]
        actual = sub.sum()
        m = len(nodes_k) * (len(nodes_k) - 1)
        all_w = W[np.triu_indices(len(W), k=1)]
        all_w_sorted = np.sort(all_w[all_w > 0])[::-1]
        max_w = all_w_sorted[:m // 2].sum() * 2
        rc.append(actual / max_w if max_w > 0 else np.nan)
    return k_levels, np.array(rc)


def compute_rich_club(sc_ctx, n_null=100):
    """Return observed RC, null distribution, and normalized RC."""
    n_ctx = sc_ctx.shape[0]
    k_levels, rc_coeff = rich_club_coefficient(sc_ctx)
    np.random.seed(42)
    rc_null = []
    upper_idx = np.triu_indices(n_ctx, k=1)
    for _ in range(n_null):
        W_null = np.zeros_like(sc_ctx)
        shuffled = sc_ctx[upper_idx].copy()
        np.random.shuffle(shuffled)
        W_null[upper_idx] = shuffled
        W_null = W_null + W_null.T
        _, rc_rand = rich_club_coefficient(W_null, k_levels)
        rc_null.append(rc_rand)
    rc_null = np.array(rc_null)
    rc_norm = rc_coeff / np.nanmean(rc_null, axis=0)
    return k_levels, rc_coeff, rc_null, rc_norm


def compute_sc_fc_coupling(sc_ctx, fc_ctx, sc_ctx_labels):
    """Return edge-level and node-level SC-FC coupling data."""
    n_ctx = sc_ctx.shape[0]
    triu = np.triu_indices(n_ctx, k=1)
    sc_edges = sc_ctx[triu]
    fc_edges = fc_ctx[triu]
    r, p = stats.pearsonr(np.log1p(sc_edges), fc_edges)

    sc_node_strength = sc_ctx.sum(axis=1)
    fc_node_strength = fc_ctx.mean(axis=1)
    r_node, p_node = stats.pearsonr(sc_node_strength, fc_node_strength)

    return sc_edges, fc_edges, r, p, sc_node_strength, fc_node_strength, r_node, p_node


def compute_degree_and_fc_strength(sc_ctx, fc_ctx):
    """Return SC node degree and FC mean strength per node."""
    sc_degree = (sc_ctx > 0).sum(axis=1)
    fc_strength = fc_ctx.mean(axis=1)
    return sc_degree, fc_strength
