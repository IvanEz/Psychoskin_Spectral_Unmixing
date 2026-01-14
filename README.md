
## Overview
This project repository contains the code to analyze Raman/reflectance spectra using constrained non-negative matrix factorization (NMF) and UMAP visualization strategies. It supports fixing known basis spectra while discovering unknown components, multimodal fusion UMAP, and weighted UMAP. Statistical comparisons between cohorts (Stress vs Control) can be generated from NMF weights.

## Project Structure
```
spectral-analysis
├── src
│   ├── perform_analysis.py          # Runs constrained NMF with fixed known components
│   ├── functions.py                 # Preprocessing, loaders, plotting, NMF helpers
│   ├── umap_visualize_multimodal.py # Multimodal UMAP (fusion across modalities)
│   ├── weighted_umap_strategy.py    # UMAP with NMF‑derived weighting masks
│   ├── strategy_utils.py            # Shared helpers for UMAP strategies
│   ├── plot.py                      # Boxplots + stats CSV for CTR vs STR NMF weights
│   └── utils.py                     # Additional utilities
├── data
│   ├── CTR/                         # Control spectra
│   ├── STR/                         # Stress spectra
│   └── substances/                  # Known substance reference spectra
├── config
│   └── config.ini                   # Parameters for perform_analysis.py
├── results                          # Generated outputs (ignored by git)
├── plots                            # Figures and stats (ignored by git)
├── requirements.txt
└── README.md
```

## Installation
Install core dependencies and the domain libraries used by the UMAP tools and preprocessing:

```bash
pip install -r requirements.txt
pip install ramanspy umap-learn pandas scipy
```

## Data & Configuration
Edit `config/config.ini` to select inputs and parameters:
- `data_dir`: sample directory (`data/STR` or `data/CTR`)
- `data_dir_known`: known substances directory (`data/substances`)
- `prefix`: file prefix for samples (e.g. `f378_STR`)
- `prefix_known`: file prefix for knowns (e.g. `f378`)
- `strategy`: preprocessing (`MinMax` or `AUC`), applied consistently to samples and knowns
- `n_unknown`: number of unknown components to discover

Input files are simple text with two columns: wavelength, intensity.

## Constrained NMF (fixed known components)
- Implementation: `functions.constrained_nmf_fixed_known(X, H_known, n_unknown)` keeps the first rows of `H` fixed to `H_known` and alternates MU updates for `W` and only the unknown rows of `H`.
- Entry point: `src/perform_analysis.py`
  - Preprocesses samples and knowns consistently, aligns spectral axes, enforces non‑negativity, normalises `H_known` rows, fits constrained NMF, and saves outputs.
  - Outputs to `results/`:
    - `..._weights_u<n>.txt` (W)
    - `..._H_components_u<n>.txt` (H)
    - `..._spectra_u<n>.png` (known + unknown components). Plots are non‑interactive (no window).

Run with the config in place:

```bash
python src/perform_analysis.py
```

For grid runs across prefixes/strategies/unknowns, use:

```bash
bash run_spectral_analysis.sh
```

## Multimodal UMAP
Script: `src/umap_visualize_multimodal.py`
- Fuses multiple modalities per subject and runs UMAP to visualise cohort separation.
- Saves plots under `results/multimodal/...` and logs a separation metric (centroid ratio) to CSV.

Examples:
```bash
# Defaults (modalities f378 f445 REFL1 REFL2 RAM, MinMax)
python src/umap_visualize_multimodal.py

# Custom modalities and fusion mode with metrics CSV path
python src/umap_visualize_multimodal.py \
  --modalities f378 RAM \
  --fusion-mode concat_stats \
  --metrics-csv results/multimodal/multimodal_separation_summary.csv
```

Key flags: `--n-neighbors`, `--min-dist`, `--metric`, `--no-scale`, `--no-annotate`.

## Weighted UMAP Strategy
Script: `src/weighted_umap_strategy.py`
- Runs NMF on fused spectra to create a soft weighting mask, then UMAP.
- Saves per‑run plots and embeddings under `results/strategies/weighted_umap/...` and appends metrics to `results/strategies/weighted_umap/metrics.csv`.

Example:
```bash
python src/weighted_umap_strategy.py \
  --preprocessing MinMax \
  --n-components 6 \
  --weight-strength 1.0 \
  --n-neighbors 8 --min-dist 0.25
```

## Batch Runner for UMAP Strategies
Script: `run_umap_strategies.sh`
- Runs all predefined modality pairs through four strategies: weighted, error‑weighted, hybrid, and supervised‑metric UMAP.
- Sweeps both preprocessing pipelines (MinMax, AUC) using shared defaults.

Run:
```bash
bash run_umap_strategies.sh
```

Environment overrides:
- `PYTHON_BIN` (default `python`), `N_COMPONENTS` (6), `N_NEIGHBORS` (8), `MIN_DIST` (0.25), `RANDOM_STATE` (42)

## Statistics on NMF Weights (CTR vs STR)
Script: `src/plot.py`
- Scans `results/*_weights_u<n>.txt` for matching CTR/STR runs; for each component, produces boxplots and performs a one‑sided Mann–Whitney U test (Stress > Control).
- Outputs:
  - Plots under `plots/`
  - Per‑run statistics CSV under `plots/stats/<prefix>_<strategy>_u<n>_stats.csv` with counts, means/medians, U, p‑value, and median differences.

Run:
```bash
python src/plot.py
```

Note: If you need two‑sided tests or FDR corrections, extend `plot.py` accordingly.

## Notes
- Plots are saved to disk without opening GUI windows. To show interactively, pass `show=True` to `plot_and_save_spectra` when calling it.

## Contacts
For questions regarding the code, please contact fatih.ozlugedik@tum.de and ivan.ezhov@tum.de
