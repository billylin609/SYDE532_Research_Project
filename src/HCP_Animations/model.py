# %% model.py — HCP_Animations: data loading and animation state computation

import numpy as np
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

from enigmatoolbox.datasets import load_sc

np.random.seed(42)

# ── Anatomical 2-D positions ──────────────────────────────────────────────────
REGION_XY = {
    'bankssts':                 (0.72, -0.02),
    'caudalanteriorcingulate':  (0.12,  0.32),
    'caudalmiddlefrontal':      (0.52,  0.50),
    'cuneus':                   (0.18, -0.82),
    'entorhinal':               (0.42, -0.62),
    'fusiform':                 (0.58, -0.60),
    'inferiorparietal':         (0.68, -0.32),
    'inferiortemporal':         (0.72, -0.50),
    'isthmuscingulate':         (0.16, -0.42),
    'lateraloccipital':         (0.62, -0.80),
    'lateralorbitofrontal':     (0.54,  0.80),
    'lingual':                  (0.30, -0.75),
    'medialorbitofrontal':      (0.20,  0.88),
    'middletemporal':           (0.78, -0.18),
    'parahippocampal':          (0.36, -0.52),
    'paracentral':              (0.22,  0.08),
    'parsopercularis':          (0.64,  0.42),
    'parsorbitalis':            (0.62,  0.76),
    'parstriangularis':         (0.60,  0.58),
    'pericalcarine':            (0.10, -0.90),
    'postcentral':              (0.58,  0.14),
    'posteriorcingulate':       (0.13, -0.22),
    'precentral':               (0.58,  0.30),
    'precuneus':                (0.22, -0.32),
    'rostralanteriorcingulate': (0.11,  0.52),
    'rostralmiddlefrontal':     (0.50,  0.66),
    'superiorfrontal':          (0.34,  0.76),
    'superiorparietal':         (0.40, -0.46),
    'superiortemporal':         (0.76,  0.08),
    'supramarginal':            (0.70, -0.10),
    'frontalpole':              (0.44,  0.96),
    'temporalpole':             (0.74,  0.42),
    'transversetemporal':       (0.70,  0.20),
    'insula':                   (0.86,  0.24),
}

BRAAK_KW = {
    1: ['entorhinal', 'parahippocampal'],
    2: ['isthmuscingulate', 'inferiortemporal', 'middletemporal', 'fusiform', 'lingual'],
    3: ['posteriorcingulate', 'precuneus', 'inferiorparietal', 'superiorparietal',
        'lateraloccipital', 'superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal'],
}

TAU_CMAP = LinearSegmentedColormap.from_list(
    'tau', ['#4e8ec9', '#ffffff', '#ffffb2', '#fd8d3c', '#bd0026', '#67000d'])

ML_CMAP = LinearSegmentedColormap.from_list(
    'mlcmap', ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b'])


def load_sc_layout():
    """Load SC data and compute anatomical layout positions."""
    sc_ctx, labels_arr, _, _ = load_sc()
    labels = list(labels_arr)
    N = sc_ctx.shape[0]

    pos = {}
    for i, lab in enumerate(labels):
        hemi, region = lab.split('_', 1)
        rx, ry = REGION_XY.get(region, (0.5, 0.0))
        sign = -1.0 if hemi == 'L' else 1.0
        pos[i] = (sign * rx, ry)

    braak_stage = np.zeros(N, dtype=int)
    for stage, kws in BRAAK_KW.items():
        for i, lab in enumerate(labels):
            if any(kw in lab.lower() for kw in kws):
                braak_stage[i] = stage

    deg = sc_ctx.sum(axis=1)
    node_size = 40 + 120 * (deg - deg.min()) / (deg.max() - deg.min() + 1e-9)

    W_MAX = sc_ctx.max()
    EDGES = [(i, j) for i in range(N) for j in range(i + 1, N) if sc_ctx[i, j] > 0]

    return sc_ctx, labels, N, pos, braak_stage, node_size, W_MAX, EDGES


def tau_trajectory(stage, frame, total_frames=45):
    """Return tau burden [0,1] for a node of given Braak stage at frame."""
    if stage == 0:
        return min(0.15, frame / total_frames * 0.15)
    windows = {1: (5, 14), 2: (15, 24), 3: (25, 38)}
    start, end = windows[stage]
    if frame < start:
        return 0.0
    if frame >= end:
        return 1.0
    return (frame - start) / (end - start)


def clinical_label(frame):
    """Return clinical phase label for a given animation frame."""
    if frame < 5:  return 'Pre-clinical (Silent)'
    if frame < 15: return 'Pre-clinical → Braak I-II  |  Entorhinal Tau Seeding'
    if frame < 25: return 'MCI / Early AD  |  Braak III-IV  |  Limbic Spread'
    if frame < 39: return 'Moderate–Severe AD  |  Braak V-VI  |  Isocortical Spread'
    return 'End-stage AD  |  Widespread Tau Pathology'


def run_motter_lai_trace(sc_ctx, seed, alpha):
    """
    Run Motter-Lai cascade and record per-round state including fractional loads.
    Returns history, load_history, C.
    """
    N_total = sc_ctx.shape[0]
    G0 = nx.from_numpy_array((sc_ctx > 0).astype(float))
    L0 = nx.betweenness_centrality(G0, normalized=False)
    C = {i: (1 + alpha) * L0[i] for i in range(N_total)}

    def frac_loads(active_nodes):
        if len(active_nodes) < 2:
            return {i: 0.0 for i in active_nodes}
        sub = sc_ctx[np.ix_(active_nodes, active_nodes)]
        Gs = nx.from_numpy_array((sub > 0).astype(float))
        Ls = nx.betweenness_centrality(Gs, normalized=False)
        out = {}
        for new_i, orig_i in enumerate(active_nodes):
            cap = C[orig_i]
            out[orig_i] = Ls[new_i] / cap if cap > 1e-9 else 0.0
        return out

    active = [i for i in range(N_total) if i != seed]
    removed = {seed}
    history = [{seed}]
    load_history = [frac_loads(active)]

    for _ in range(N_total):
        if len(active) < 2:
            break
        fl = frac_loads(active)
        overloaded = {orig_i for orig_i, frac in fl.items() if frac > 1.0}
        if not overloaded:
            break
        removed.update(overloaded)
        history.append(overloaded)
        active = [i for i in active if i not in overloaded]
        load_history.append(frac_loads(active))

    return history, load_history, C


def build_ml_frame_info(n_rounds):
    """Build frame metadata for the Motter-Lai animation."""
    PHASE_A = 4
    PHASE_B = 3
    ROUND_FRAMES = 4
    PHASE_D = 5

    frames_info = []
    for f in range(PHASE_A):
        frames_info.append({'phase': 'A', 'round': -1, 'sub': f})
    for f in range(PHASE_B):
        frames_info.append({'phase': 'B', 'round': 0, 'sub': f})
    for r in range(1, n_rounds):
        for f in range(ROUND_FRAMES):
            frames_info.append({'phase': 'C', 'round': r, 'sub': f})
    for f in range(PHASE_D):
        frames_info.append({'phase': 'D', 'round': n_rounds, 'sub': f})

    return frames_info, ROUND_FRAMES
