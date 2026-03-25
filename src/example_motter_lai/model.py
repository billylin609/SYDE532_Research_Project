# %% model.py — example_motter_lai: 12-node network setup and NDLib cascade computation

import numpy as np
import networkx as nx

import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

np.random.seed(42)

# ── Network: 12-node hub-provincial-nonhub-leaf structure ─────────────────────
EDGES = [
    (0,1),(0,2),(0,3),(0,4),(0,5),  # node 0: central hub
    (1,6),(1,7),                     # node 1: provincial hub A
    (2,7),(2,8),                     # node 2: provincial hub B
    (3,9),                           # node 3: non-hub connector
    (4,9),(4,10),                    # node 4: non-hub connector
    (5,10),(5,11),                   # node 5: non-hub connector
    (6,7),(8,9),(10,11),             # leaf cross-links
]
ROLES = {
    0:'Hub',
    1:'Provincial Hub A', 2:'Provincial Hub B',
    3:'Non-Hub Connector', 4:'Non-Hub Connector', 5:'Non-Hub Connector',
    6:'Leaf', 7:'Leaf', 8:'Leaf', 9:'Leaf', 10:'Leaf', 11:'Leaf'
}
POS = {
    0:(0.0, 0.0),
    1:(-1.6, 0.9),  2:(-1.6,-0.9),
    3:(0.0, 1.9),   4:(1.6, 0.9),  5:(1.6,-0.9),
    6:(-2.6, 1.6),  7:(-2.6, 0.2),
    8:(-2.6,-1.6),  9:(0.0, 2.9),
    10:(2.6, 1.6),  11:(2.6,-0.5),
}
N = 12

ROLE_COLORS = {
    'Hub':               '#ff6600',
    'Provincial Hub A':  '#f39c12',
    'Provincial Hub B':  '#f39c12',
    'Non-Hub Connector': '#27ae60',
    'Leaf':              '#3498db',
}

ALPHA = 0.3   # tolerance: C_j = (1+alpha) * L_j(0)


def build_network():
    """Build the 12-node NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(EDGES)
    return G


def compute_loads_capacities(G):
    """Compute initial betweenness loads and capacities."""
    L0 = nx.betweenness_centrality(G, normalized=False)
    C = {n: (1 + ALPHA) * L0[n] for n in range(N)}
    return L0, C


def run_ndlib_cascade(G, C):
    """
    Run NDLib ThresholdModel cascade seeded at node 0 (hub attack).
    Returns list of round dicts with 'status' keys.
    """
    model = ep.ThresholdModel(G)
    cfg = mc.Configuration()
    cfg.add_model_parameter('fraction_infected', 0)

    max_C = max(C.values()) if max(C.values()) > 0 else 1.0
    for n in range(N):
        deg = G.degree(n)
        thresh = min(0.99, C[n] / (max_C * (deg + 1e-9)))
        cfg.add_node_configuration('threshold', n, thresh)

    model.set_initial_status(cfg)
    model.status[0] = 1
    model.initial_status[0] = 1

    ndlib_iterations = [{'status': dict(model.status)}]
    for _ in range(20):
        it = model.iteration()
        ndlib_iterations.append({'status': dict(model.status)})
        if sum(1 for v in it['status'].values() if v == 1) == 0:
            break

    # Deduplicate: keep only rounds where something changed
    rounds = [ndlib_iterations[0]]
    for it in ndlib_iterations[1:]:
        prev_failed = set(n for n, s in rounds[-1]['status'].items() if s == 1)
        curr_failed = set(n for n, s in it['status'].items() if s == 1)
        if curr_failed != prev_failed:
            rounds.append(it)
        if len(curr_failed) == N:
            break

    return rounds


def build_frame_data(rounds):
    """Build animation frame metadata."""
    HOLD = 6
    TRANS = 3

    frame_data = []
    for ri in range(len(rounds)):
        for f in range(HOLD):
            frame_data.append((ri, f, False))
        if ri < len(rounds) - 1:
            for f in range(TRANS):
                frame_data.append((ri, f, True))

    return frame_data, HOLD, TRANS
