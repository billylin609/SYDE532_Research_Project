============================================================
LOCAL CENTRALITY  (top 15 by Eigenvector)
============================================================
                Region  Degree  Betweenness  Closeness  Eigenvector
    R_superiorparietal  0.6119       0.0226     1.0000       0.2333
     R_superiorfrontal  0.5075       0.0014     0.9528       0.2166
     L_superiorfrontal  0.4925       0.0063     0.9118       0.2090
    L_superiorparietal  0.5224       0.0063     0.8921       0.2076
              L_insula  0.5672       0.0412     0.6845       0.1920
              R_insula  0.5672       0.0520     0.7059       0.1837
           R_precuneus  0.4627       0.0118     0.7624       0.1741
L_rostralmiddlefrontal  0.4478       0.0217     0.7380       0.1731
           L_precuneus  0.4179       0.0081     0.7513       0.1715
R_rostralmiddlefrontal  0.3881       0.0077     0.7135       0.1550
    L_isthmuscingulate  0.4478       0.0416     0.5271       0.1473
    R_isthmuscingulate  0.4478       0.0488     0.5151       0.1459
          L_precentral  0.3433       0.0109     0.6283       0.1422
          R_precentral  0.2836       0.0018     0.6493       0.1328
             L_lingual  0.3433       0.0041     0.4798       0.1326

============================================================
BROKER NODES  (10 identified — high betweenness, low degree)
============================================================
                   Region  Degree_rank  Betweenness_rank  Broker_score
        L_parahippocampal           32                 2        0.5506
    R_medialorbitofrontal           22                 1        0.5135
        R_parahippocampal           26                 4        0.4558
             L_entorhinal           61                11        0.3653
           R_temporalpole           46                10        0.3605
R_caudalanteriorcingulate           18                 7        0.2661
   L_lateralorbitofrontal           13                 6        0.2513
L_caudalanteriorcingulate           37                11        0.2302
       R_isthmuscingulate            9                 4        0.2126
            R_frontalpole           53                16        0.1587

============================================================
GLOBAL CENTRALITY
============================================================
  Global degree centrality               0.305970
  Global betweenness centrality          0.014859
  Global closeness centrality (mean)     4.542958
  Closeness centralization C̄ᶜ           2.871623
  Graph density                          0.305970
  Spectral radius (λ₁)                   173.536637
  Avg clustering coefficient             0.340689

1. Degree Centrality (model.py:22-24)
nx.degree_centrality(G) — unweighted. For each node, it's the fraction of other nodes it connects to: deg(v) / (n-1). Edge weights are ignored entirely.

2. Betweenness Centrality (model.py:27-29)
nx.betweenness_centrality(G, weight='weight', normalized=True) — counts how often a node lies on the shortest weighted path between all other node pairs. With weight='weight', NetworkX uses weights as distances (lower = shorter), which is backwards for fiber-count SC matrices where higher weight = stronger connection. This is likely a bug — it should use distance as an inverse weight, similar to what closeness does.

3. Closeness Centrality (model.py:32-37)
Manually inverts weights first: distance = 1 / (weight + ε), then calls nx.closeness_centrality(G_close, distance='distance'). This correctly treats high SC weight as short distance. The formula is (n-1) / sum_of_distances_to_all_others. The final result is normalized after computing the inverse distance shortest-path.

4. Eigenvector Centrality (model.py:40-42)
nx.eigenvector_centrality_numpy(G, weight='weight') — solves the principal eigenvector of the adjacency matrix. A node's score is proportional to the sum of its neighbors' scores. High-weight edges amplify influence.

Broker in the system: high betweenness centrality but low degree centrality