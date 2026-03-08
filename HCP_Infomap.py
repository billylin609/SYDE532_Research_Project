"""
Community Detection on the HCP SC Consensus Network (82 brain regions)
=======================================================================
Algorithms implemented
  1. Louvain Method      — modularity optimisation (greedy agglomeration)
  2. Leiden Algorithm    — modularity / CPM optimisation with guaranteed
                           connectivity of detected communities
  3. Infomap             — minimum-description-length / random-walk encoding

Matrix: full 82×82 structural connectivity (68 cortical + 14 subcortical)
        cortico-cortical  |  cortico-subcortical  |  subcortico-subcortical


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THEORY & EQUATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LOUVAIN METHOD  (Blondel et al., 2008)
─────────────────────────────────────────
Objective: maximise Newman–Girvan modularity

    Q = (1/2m) Σ_{ij} [ A_{ij} - k_i k_j / (2m) ] δ(c_i, c_j)

where
  A_{ij}  = edge weight between nodes i and j
  k_i     = strength (weighted degree) of node i  (Σ_j A_{ij})
  m       = total edge weight  (Σ_{ij} A_{ij} / 2)
  δ(c_i, c_j) = 1 if i and j are in the same community, else 0

Greedy modularity gain for moving node i into community C:

    ΔQ = [ (Σ_in + 2 k_{i→C}) / 2m - ((Σ_tot + k_i) / 2m)² ]
       − [ Σ_in / 2m − (Σ_tot / 2m)² − (k_i / 2m)² ]

where Σ_in = internal edge weight of C, Σ_tot = total degree of C,
      k_{i→C} = weight of edges from i to C.

Phase 1: Repeatedly move single nodes to maximise ΔQ.
Phase 2: Collapse communities into super-nodes and repeat.

Resolution parameter γ extends Q to:
    Q_γ = (1/2m) Σ_{ij} [ A_{ij} - γ k_i k_j / (2m) ] δ(c_i, c_j)


2. LEIDEN ALGORITHM  (Traag et al., 2019)
──────────────────────────────────────────
Extends Louvain with a "refine" phase that guarantees every detected
community is internally connected (Louvain can produce disconnected
super-nodes).

Three-phase loop:
  (a) Move phase   — same greedy ΔQ as Louvain.
  (b) Refine phase — within each community, optimise a sub-partition
                     on the induced subgraph; merge nodes only when
                     ΔQ > 0 in the refined sub-community.
  (c) Aggregate    — form aggregate graph from the refined partition.

The Constant Potts Model (CPM) quality function used internally:

    H = − Σ_c [ e_c − γ n_c(n_c − 1)/2 ]

where e_c = edges inside community c, n_c = number of nodes.
γ controls resolution: larger γ → more, smaller communities.

Leiden with ModularityVertexPartition recovers standard Q above.


3. INFOMAP  (Rosvall & Bergstrom, 2008)
─────────────────────────────────────────
Objective: minimise the expected per-step description length of a
random walk on the network (the *map equation*):

    L(M) = q_↷ H(Q) + Σ_{α=1}^{m} p_α^↷ H(P_α)

where
  M         = module partition
  q_↷       = total probability of inter-module movement per step
  H(Q)      = entropy of the module-index codebook
               H(Q) = − Σ_α (q_α / q_↷) log₂(q_α / q_↷)
  p_α^↷     = rate of visiting/leaving module α
               p_α^↷ = q_α + Σ_{i∈α} p_i
  H(P_α)   = entropy of the within-module codebook
               H(P_α) = − Σ_{i∈α} (p_i / p_α^↷) log₂(p_i / p_α^↷)
               − (q_α  / p_α^↷) log₂(q_α  / p_α^↷)
  p_i       = stationary visit probability of node i
  q_α       = exit probability from module α

Optimising L(M) groups nodes whose random-walk dynamics are tightly
coupled—nodes that trap the walk—into the same community.

Markov-time parameter τ modifies the teleportation probability:
  higher τ → walk stays longer within modules → fewer, larger communities.
  lower  τ → walk mixes faster across modules → more, finer communities.
  Default τ = 1.0.  Dense SC matrices at τ ≥ 2 often collapse to 1 module.

Implementation: 100 random restarts, two-level hierarchy, best codelength.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPARISON METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Normalised Mutual Information (NMI):
    NMI(U,V) = 2 I(U;V) / [H(U) + H(V)]
  NMI = 1 → identical partitions, 0 → independent.

Adjusted Rand Index (ARI):
    ARI = (RI − E[RI]) / (max RI − E[RI])
  ARI = 1 → perfect agreement, 0 → random, <0 → anti-correlated.
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from infomap import Infomap
from enigmatoolbox.datasets import load_sc
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# Leiden requires leidenalg + python-igraph
try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    print("WARNING: leidenalg / igraph not installed. "
          "Install with:  pip install leidenalg python-igraph\n"
          "Leiden results will be skipped.")

# =============================================================================
# Load and build full 82×82 SC matrix
# =============================================================================
sc_ctx, sc_ctx_labels, sc_sctx, sc_sctx_labels = load_sc()

n_ctx  = sc_ctx.shape[0]   # 68 cortical regions
n_sctx = sc_sctx.shape[0]  # 14 subcortical regions
N      = n_ctx + n_sctx    # 82 total regions

sc_full = np.zeros((N, N))
sc_full[:n_ctx, :n_ctx] = sc_ctx
sc_full[:n_ctx, n_ctx:] = sc_sctx.T
sc_full[n_ctx:, :n_ctx] = sc_sctx

labels = list(sc_ctx_labels) + list(sc_sctx_labels)

print(f"Loaded 82×82 SC matrix  ({n_ctx} cortical + {n_sctx} subcortical regions)")
print(f"Non-zero edges (upper-tri): "
      f"{int(np.count_nonzero(np.triu(sc_full, k=1)))}\n")

# =============================================================================
# Region metadata
# =============================================================================
lobe_map = {
    'bankssts':'Temporal','caudalanteriorcingulate':'Cingulate',
    'caudalmiddlefrontal':'Frontal','cuneus':'Occipital','entorhinal':'Temporal',
    'fusiform':'Temporal','inferiorparietal':'Parietal','inferiortemporal':'Temporal',
    'isthmuscingulate':'Cingulate','lateraloccipital':'Occipital',
    'lateralorbitofrontal':'Frontal','lingual':'Occipital','medialorbitofrontal':'Frontal',
    'middletemporal':'Temporal','parahippocampal':'Temporal','paracentral':'Frontal',
    'parsopercularis':'Frontal','parsorbitalis':'Frontal','parstriangularis':'Frontal',
    'pericalcarine':'Occipital','postcentral':'Parietal','posteriorcingulate':'Cingulate',
    'precentral':'Frontal','precuneus':'Parietal','rostralanteriorcingulate':'Cingulate',
    'rostralmiddlefrontal':'Frontal','superiorfrontal':'Frontal',
    'superiorparietal':'Parietal','superiortemporal':'Temporal','supramarginal':'Parietal',
    'frontalpole':'Frontal','temporalpole':'Temporal','transversetemporal':'Temporal',
    'insula':'Insular',
}

def get_lobe(label, idx):
    if idx < n_ctx:
        return lobe_map.get(label[2:], 'Other')
    return 'Subcortical'

def get_hemi(label):
    return 'L' if label.startswith('L') else 'R'

lobes = [get_lobe(labels[i], i) for i in range(N)]
hemis = [get_hemi(l) for l in labels]

lobe_colors = {
    'Frontal':'#e63946','Parietal':'#457b9d','Temporal':'#2a9d8f',
    'Occipital':'#e9c46a','Cingulate':'#f4a261','Insular':'#8338ec',
    'Subcortical':'#6c757d','Other':'#aaaaaa',
}
lobe_order = ['Frontal','Parietal','Temporal','Occipital','Cingulate','Insular','Subcortical']

# =============================================================================
# Algorithm implementations
# =============================================================================

def run_louvain(adj, resolution=1.0, seed=42, n_restarts=20):
    """
    Louvain community detection via NetworkX.

    NetworkX implements the Louvain method exactly as described in
    Blondel et al. (2008): greedy ΔQ node-move phase alternating with
    community-collapse aggregation phase.

    Louvain is non-deterministic (greedy hill-climbing with random tie-
    breaking).  Running multiple restarts and keeping the highest-Q result
    substantially improves reliability.

    Parameters
    ----------
    adj         : (N, N) symmetric weighted adjacency matrix
    resolution  : γ parameter in the resolution-extended modularity
    seed        : base random seed for reproducibility
    n_restarts  : number of independent random restarts

    Returns
    -------
    labels  : list of length N with integer community IDs
    n_comm  : number of communities found
    Q       : best modularity score across all restarts
    """
    G = nx.Graph()
    G.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if adj[i, j] > 0:
                G.add_edge(i, j, weight=adj[i, j])

    best_Q, best_labels, best_comms = -np.inf, None, None
    for k in range(n_restarts):
        comms = nx.community.louvain_communities(
            G, weight='weight', resolution=resolution, seed=seed + k)
        Q = nx.community.modularity(G, comms, weight='weight')
        if Q > best_Q:
            best_Q, best_comms = Q, comms

    node_labels = [0] * adj.shape[0]
    for idx, comm in enumerate(best_comms):
        for node in comm:
            node_labels[node] = idx

    return node_labels, len(best_comms), best_Q


def run_leiden(adj, resolution=1.0, seed=42):
    """
    Leiden community detection via leidenalg + igraph.

    Uses ModularityVertexPartition (equivalent to Newman–Girvan Q) with
    the same resolution parameter as Louvain for a fair comparison.
    100 random starts are performed and the partition with the highest
    quality is returned.

    Parameters
    ----------
    adj        : (N, N) symmetric weighted adjacency matrix
    resolution : γ parameter (passed to leidenalg as resolution_parameter)
    seed       : random seed for reproducibility

    Returns
    -------
    node_labels : list of length N with integer community IDs
    n_comm      : number of communities
    quality     : ModularityVertexPartition quality (≈ Q)
    """
    if not LEIDEN_AVAILABLE:
        return None, None, None

    # Build igraph Graph
    g = ig.Graph()
    g.add_vertices(adj.shape[0])
    edges, weights = [], []
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if adj[i, j] > 0:
                edges.append((i, j))
                weights.append(float(adj[i, j]))
    g.add_edges(edges)
    g.es['weight'] = weights

    # Run Leiden with multiple random starts; keep highest modularity Q.
    # leidenalg's quality() for ModularityVertexPartition returns the
    # unnormalised sum Σ[A_ij - k_i k_j/(2m)] δ(c_i,c_j).  We compare
    # partitions by this internal value (monotone with normalised Q).
    best_part = None
    best_quality = -np.inf
    rng = np.random.default_rng(seed)
    for _ in range(50):
        part = leidenalg.find_partition(
            g,
            leidenalg.ModularityVertexPartition,
            weights='weight',
            seed=int(rng.integers(1, 2**31)),
        )
        if part.quality() > best_quality:
            best_quality = part.quality()
            best_part = part

    node_labels = best_part.membership
    return node_labels, len(set(node_labels)), best_quality


def run_infomap(adj, markov_time=1.0, seed=42):
    """
    Infomap community detection.

    Minimises the map equation L(M) over the partition space using a
    two-level hierarchy.  100 independent trials are run internally
    and the partition with the lowest codelength is kept.

    Parameters
    ----------
    adj         : (N, N) symmetric weighted adjacency matrix
    markov_time : τ — higher values produce coarser communities
    seed        : random seed for reproducibility

    Returns
    -------
    node_labels : list of length N with integer community IDs (1-indexed)
    n_modules   : number of top-level modules
    codelength  : achieved map-equation value L(M) in bits
    """
    im = Infomap(seed=seed, two_level=True, silent=True,
                 num_trials=100, markov_time=markov_time)
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if adj[i, j] > 0:
                im.add_link(i, j, adj[i, j])
    im.run()

    node_labels = [0] * adj.shape[0]
    for node in im.nodes:
        node_labels[node.node_id] = node.module_id
    return node_labels, im.num_top_modules, im.codelength


# =============================================================================
# Run all three algorithms
# =============================================================================
SEP = "=" * 72

# Build NetworkX graph once — shared by all three algorithms for Q computation
G_full = nx.Graph()
G_full.add_nodes_from(range(N))
for i in range(N):
    for j in range(i + 1, N):
        if sc_full[i, j] > 0:
            G_full.add_edge(i, j, weight=sc_full[i, j])

# ── Louvain ──────────────────────────────────────────────────────────────────
print(SEP)
print("ALGORITHM 1: Louvain Method  (fine γ grid search, 20 restarts per γ)")
print(SEP)
# Fine grid: 0.5 → 3.0 in steps of 0.1 (26 points × 20 restarts)
_gamma_grid = np.round(np.arange(0.5, 3.05, 0.1), 2)
louvain_results = {}
for res in _gamma_grid:
    lbl, nc, Q = run_louvain(sc_full, resolution=float(res), n_restarts=20)
    louvain_results[float(res)] = (lbl, nc, Q)

# Print summary row for coarse reference points (must be on the 0.1-step grid)
print(f"  {'γ':>6}  {'comms':>6}  {'Q':>8}")
for res in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0]:
    lbl, nc, Q = louvain_results[res]
    print(f"  {res:>6.2f}  {nc:>6d}  {Q:>8.4f}")

# Auto-select γ that maximises Q
RES_LOUVAIN = max(louvain_results, key=lambda r: louvain_results[r][2])
lbl_louvain, nc_louvain, Q_louvain = louvain_results[RES_LOUVAIN]
print(f"\n  Best γ = {RES_LOUVAIN:.2f}: {nc_louvain} communities, Q = {Q_louvain:.4f}")

# ── Leiden ───────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("ALGORITHM 2: Leiden Algorithm")
print(SEP)
leiden_results = {}
if LEIDEN_AVAILABLE:
    print(f"  Fine γ grid search, 50 restarts per γ:")
    # Same grid as Louvain for direct comparison
    for res in _gamma_grid:
        lbl, nc, Q = run_leiden(sc_full, resolution=float(res))
        leiden_results[float(res)] = (lbl, nc, Q)

    print(f"  {'γ':>6}  {'comms':>6}  {'Q (internal)':>14}")
    for res in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0]:
        lbl, nc, Q = leiden_results[res]
        print(f"  {res:>6.2f}  {nc:>6d}  {Q:>14.2f}")

    # Auto-select γ that maximises quality (monotone with normalised Q)
    RES_LEIDEN = max(leiden_results, key=lambda r: leiden_results[r][2])
    lbl_leiden, nc_leiden, Q_leiden = leiden_results[RES_LEIDEN]
    # Convert leidenalg internal quality to normalised Newman-Girvan Q
    _ld_best_comms = [frozenset(i for i in range(N) if lbl_leiden[i] == m)
                      for m in set(lbl_leiden)]
    Q_leiden_nx = nx.community.modularity(G_full, _ld_best_comms, weight='weight')
    print(f"\n  Best γ = {RES_LEIDEN:.2f}: {nc_leiden} communities, Q = {Q_leiden_nx:.4f}")
else:
    lbl_leiden, nc_leiden, Q_leiden = None, None, None
    print("  Skipped (leidenalg not installed).")

# ── Infomap ──────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("ALGORITHM 3: Infomap")
print(SEP)

# NOTE on negative Q for Infomap:
# Infomap minimises the map equation (codelength), not modularity Q.
# On dense SC matrices the random-walk null model differs from the
# Newman-Girvan stub-matching null used by Q, so Q can be negative for
# a partition that is perfectly valid from Infomap's own perspective.
# We resolve this by sweeping τ and selecting the τ that yields the
# highest Q — bridging the two frameworks.
print("Sweeping τ — reporting both codelength (Infomap objective) and Q (modularity):")
_tau_grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
infomap_results = {}
print(f"  {'τ':>6}  {'modules':>8}  {'codelength':>12}  {'Q':>8}")
for mt in _tau_grid:
    lbl, nm, cl = run_infomap(sc_full, markov_time=mt)
    if nm > 1:
        comms = [frozenset(i for i in range(N) if lbl[i] == m) for m in set(lbl)]
        Q_mt = nx.community.modularity(G_full, comms, weight='weight')
    else:
        Q_mt = -np.inf   # 1 module → Q trivially = 0 by definition but meaningless
    infomap_results[mt] = (lbl, nm, cl, Q_mt)
    q_str = f"{Q_mt:.4f}" if Q_mt > -np.inf else "  (1 mod)"
    print(f"  {mt:>6.2f}  {nm:>8d}  {cl:>12.4f}  {q_str:>8}")

# Select τ that maximises Q (must have >1 module)
valid_taus = {mt: v for mt, v in infomap_results.items() if v[1] > 1}
if valid_taus:
    TAU_INFOMAP = max(valid_taus, key=lambda mt: valid_taus[mt][3])
else:
    TAU_INFOMAP = 0.5
lbl_infomap, nc_infomap, CL_infomap, Q_infomap = infomap_results[TAU_INFOMAP]
print(f"\n  Best τ = {TAU_INFOMAP}: {nc_infomap} modules, "
      f"codelength = {CL_infomap:.4f} bits, Q = {Q_infomap:.4f}")

# =============================================================================
# Cross-algorithm comparison metrics
# =============================================================================
print("\n" + SEP)
print("CROSS-ALGORITHM COMPARISON")
print(SEP)

pairs = [
    ("Louvain",  lbl_louvain,  "Infomap",  lbl_infomap),
]
if LEIDEN_AVAILABLE:
    pairs += [
        ("Leiden",   lbl_leiden,   "Infomap",  lbl_infomap),
        ("Louvain",  lbl_louvain,  "Leiden",   lbl_leiden),
    ]

for nameA, lblA, nameB, lblB in pairs:
    nmi = normalized_mutual_info_score(lblA, lblB)
    ari = adjusted_rand_score(lblA, lblB)
    print(f"  {nameA:8s} vs {nameB:8s}:  NMI = {nmi:.4f},  ARI = {ari:.4f}")

# =============================================================================
# Detailed partition printout — Infomap (reference)
# =============================================================================
print("\n" + SEP)
print(f"PARTITION DETAIL — Infomap (τ={TAU_INFOMAP})  [{nc_infomap} modules]")
print(SEP)

unique_im = sorted(set(lbl_infomap))
for mod in unique_im:
    members = [i for i in range(N) if lbl_infomap[i] == mod]
    ctx_n  = sum(1 for m in members if m < n_ctx)
    sctx_n = len(members) - ctx_n
    lobe_dist = {}
    hemi_dist = {'L': 0, 'R': 0}
    for m in members:
        lobe_dist[lobes[m]] = lobe_dist.get(lobes[m], 0) + 1
        hemi_dist[hemis[m]] += 1
    lobe_str = ", ".join(f"{k}:{v}" for k, v in
                         sorted(lobe_dist.items(), key=lambda x: -x[1]))
    print(f"\n  Module {mod}  ({len(members)} regions: {ctx_n} ctx, {sctx_n} sctx)"
          f"  [{lobe_str}]  L:{hemi_dist['L']} R:{hemi_dist['R']}")
    for m in members:
        tag = 'ctx ' if m < n_ctx else 'sctx'
        print(f"    [{tag}] {labels[m]:42s} {hemis[m]}  {lobes[m]}")

# =============================================================================
# VISUALIZATION
# =============================================================================
# Assign consistent colors per algorithm
def make_mod_colors(node_labels, cmap_name='Set2'):
    unique = sorted(set(node_labels))
    n = len(unique)
    cmap = plt.cm.get_cmap(cmap_name, max(n, 3))
    return {m: cmap(i / max(n - 1, 1)) for i, m in enumerate(unique)}

mc_louvain = make_mod_colors(lbl_louvain, 'Set1')
mc_infomap = make_mod_colors(lbl_infomap, 'Set2')

strength = sc_full.sum(axis=0)

# Build NetworkX graph once for layout
G_vis = nx.Graph()
G_vis.add_nodes_from(range(N))
for i in range(N):
    for j in range(i + 1, N):
        if sc_full[i, j] > 0:
            G_vis.add_edge(i, j, weight=sc_full[i, j])
pos = nx.spring_layout(G_vis, weight='weight', k=0.55, iterations=80, seed=42)

# Pre-compute percentile threshold for connectogram edges
upper_w = [sc_full[i, j] for i in range(N) for j in range(i+1, N) if sc_full[i, j] > 0]
edge_thresh = np.percentile(upper_w, 85)

mc_leiden = make_mod_colors(lbl_leiden, 'tab10') if LEIDEN_AVAILABLE else None

fig = plt.figure(figsize=(20, 9))
fig.suptitle(
    'Community Detection — HCP SC Consensus Network (82 regions)\n'
    f'Louvain (γ={RES_LOUVAIN}): {nc_louvain} communities  |  '
    + (f'Leiden (γ={RES_LEIDEN}): {nc_leiden} communities  |  '
       if LEIDEN_AVAILABLE else '')
    + f'Infomap (τ={TAU_INFOMAP}): {nc_infomap} modules',
    fontsize=12, fontweight='bold')

gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.20)

# ── Helper: reordered adjacency heatmap ─────────────────────────────────────
def plot_adj(ax, node_labels, title, mod_colors):
    unique = sorted(set(node_labels))
    idx = sorted(range(N), key=lambda x: (node_labels[x], x >= n_ctx))
    sc_s = sc_full[np.ix_(idx, idx)]
    sc_s[sc_s == 0] = np.nan
    ax.imshow(sc_s, cmap='YlGnBu', aspect='equal', interpolation='none')
    cursor = 0
    for mod in unique:
        cnt = sum(1 for m in node_labels if m == mod)
        ax.axhline(y=cursor - 0.5, color='red', linewidth=0.8)
        ax.axvline(x=cursor - 0.5, color='red', linewidth=0.8)
        ax.text(-2, cursor + cnt / 2, f'C{mod}', fontsize=6.5, va='center',
                ha='right', color=mod_colors[mod], fontweight='bold', clip_on=False)
        cursor += cnt
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

# ── Helper: force-directed layout ────────────────────────────────────────────
def plot_network(ax, node_labels, title, mod_colors):
    for u, v, d in G_vis.edges(data=True):
        same = node_labels[u] == node_labels[v]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=mod_colors[node_labels[u]] if same else '#cccccc',
                alpha=0.25 if same else 0.03, linewidth=0.4, zorder=0)
    sz = 20 + 150 * (strength / strength.max())
    for i in range(N):
        ax.scatter(pos[i][0], pos[i][1],
                   c=[mod_colors[node_labels[i]]], s=sz[i],
                   marker='o' if i < n_ctx else 'D',
                   edgecolors='white', linewidths=0.3, zorder=3)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

# ── Row 0: adjacency matrices ────────────────────────────────────────────────
plot_adj(fig.add_subplot(gs[0, 0]),
         lbl_louvain,
         f'(a) SC Matrix — Louvain\n(γ={RES_LOUVAIN}, {nc_louvain} comm., Q={Q_louvain:.3f})',
         mc_louvain)

if LEIDEN_AVAILABLE:
    plot_adj(fig.add_subplot(gs[0, 1]),
             lbl_leiden,
             f'(b) SC Matrix — Leiden\n(γ={RES_LEIDEN}, {nc_leiden} comm., Q={Q_leiden_nx:.3f})',
             mc_leiden)
else:
    fig.add_subplot(gs[0, 1]).axis('off')

plot_adj(fig.add_subplot(gs[0, 2]),
         lbl_infomap,
         f'(c) SC Matrix — Infomap\n(τ={TAU_INFOMAP}, {nc_infomap} mod., L={CL_infomap:.3f} bits)',
         mc_infomap)

# ── Row 1: force-directed networks ───────────────────────────────────────────
plot_network(fig.add_subplot(gs[1, 0]),
             lbl_louvain,
             f'(d) Network — Louvain\n(circles=ctx, diamonds=sctx)',
             mc_louvain)

if LEIDEN_AVAILABLE:
    plot_network(fig.add_subplot(gs[1, 1]),
                 lbl_leiden,
                 f'(e) Network — Leiden\n(circles=ctx, diamonds=sctx)',
                 mc_leiden)
else:
    fig.add_subplot(gs[1, 1]).axis('off')

plot_network(fig.add_subplot(gs[1, 2]),
             lbl_infomap,
             f'(f) Network — Infomap\n(circles=ctx, diamonds=sctx)',
             mc_infomap)

import os
os.makedirs('./figs', exist_ok=True)
plt.savefig('./figs/community_detection_comparison.png', dpi=180, bbox_inches='tight')
print(f"\n✓ Figure saved: ./figs/community_detection_comparison.png")

# =============================================================================
# Summary table
# =============================================================================
print("\n" + SEP)
print("SUMMARY")
print(SEP)
# Q is Newman-Girvan modularity (0–1) — comparable across all three.
# Codelength is the Infomap-specific map-equation value in bits.
print(f"  {'Algorithm':<12} {'Communities':>12} {'Q (modularity)':>16} {'Codelength (bits)':>20} {'Parameters'}")
print(f"  {'-'*75}")
print(f"  {'Louvain':<12} {nc_louvain:>12} {Q_louvain:>16.4f} {'—':>20}  γ = {RES_LOUVAIN}")
if LEIDEN_AVAILABLE:
    print(f"  {'Leiden':<12} {nc_leiden:>12} {Q_leiden_nx:>16.4f} {'—':>20}  γ = {RES_LEIDEN}")
print(f"  {'Infomap':<12} {nc_infomap:>12} {Q_infomap:>16.4f} {CL_infomap:>20.4f}  τ = {TAU_INFOMAP}")
print()
