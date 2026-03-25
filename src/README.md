# SYDE532 Project — Source Code

## Run order
Each script is standalone (loads data via ENIGMA `load_sc()`). Run from inside `src/`:

```bash
cd src
python HCP_Visualize.py          # Dataset overview
python HCP_Centrality.py         # Centrality analysis
python HCP_NeuralNetwork.py      # Network topology + efficiency-resilience
python HCP_Community.py          # Community detection
python HCP_Perturbation.py       # Node addition/removal simulation
python HCP_MotterLai.py          # Motter-Lai cascade failure
python HCP_AlzheimerComparison.py  # AD biology vs Motter-Lai
python HCP_Animations.py         # Animated GIFs
python HCP_AD_Visualization.py   # AD social impact + simple demo
```

All figures saved to `../figs/`.

## Prerequisites
```bash
pip install -r ../requirements.txt
cd ../ENIGMA && pip install .
```

## Annotation files
Each `.py` has a matching `.md` with concept explanations, output descriptions, and references.
