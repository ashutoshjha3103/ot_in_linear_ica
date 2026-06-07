# Optimal Transport for Linear Independent Component Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Master's thesis project — TPM UAI 2026 workshop paper**  
**Author:** Ashutosh Jha  
**Program:** M.Sc. Quantitative Data Science Methods  
**Affiliations:** Eberhard Karls University of Tübingen & Empirical Inference Dept., Max Planck Institute for Intelligent Systems

---

## Overview

**OT-ICA** recovers independent source signals from linear mixtures by maximizing the 1D squared Wasserstein distance ($W_2^2$) between each projected component and a standard Gaussian — computed exactly by quantile sorting, without density estimation or parametric assumptions.

Classical algorithms (FastICA, JADE, InfoMax) replace the true non-Gaussianity measure with surrogates (logcosh, fourth-order cumulants, parametric log-likelihoods) that fail on heterogeneous source distributions. OT-ICA bypasses this by using an exact, assumption-free contrast function. Optimization runs on the Orthogonal Group via Riemannian gradient ascent with symmetric-decorrelation retraction.

The core library is in `src/wasserstein_ica/`. Experimental notebooks are in `exp/`. The master thesis and UAI workshop paper are in `report/` and `uai_workshop/` respectively.

---

## Requirements

- **Python** 3.8 or newer
- Runtime dependencies are declared in [`pyproject.toml`](pyproject.toml): NumPy, SciPy, PyTorch (≥ 1.10), POT, scikit-learn, pandas, matplotlib, seaborn, tqdm, joblib, Jupyter.
- EEG application additionally requires `mne` (listed in [`exp/requirements.txt`](exp/requirements.txt)).

---

## Installation

```bash
git clone https://github.com/ashutoshjha3103/OT_IN_LINEAR_ICA.git
cd OT_IN_LINEAR_ICA

python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install --upgrade pip setuptools wheel
pip install -e .
pip install -r exp/requirements.txt   # for EEG notebook
```

Verify:
```bash
python -c "from wasserstein_ica import WassersteinICA; print('OK')"
```

---

## Quick start

```python
import numpy as np
import torch
from wasserstein_ica import WassersteinICA

rng = np.random.default_rng(0)
S = rng.laplace(0, 1, size=(3, 2000))          # 3 Laplace sources
A = rng.standard_normal((3, 3))
X = torch.tensor(A @ S, dtype=torch.float32)

ica = WassersteinICA(X)
ica.whiten()
W_est, w2_score = ica.optimize_wasserstein2(continuous=True)

print("Estimated unmixing row:", W_est)
print("W2² objective:         ", float(w2_score))
```

---

## Repository layout

```
OT_IN_LINEAR_ICA/
├── src/
│   └── wasserstein_ica/       # WassersteinICA — core optimization and contrast functions
├── exp/                       # Experiments and figures (run from here)
│   ├── OT-ICA_Methodology_Validation.ipynb         # [1] Sanity check — start here
│   ├── OT-ICA_W1_vs_W2.ipynb                       # [2] Why W2 over W1
│   ├── OT-ICA_stiefel_vs_FastICA_scaling.ipynb     # [3] Scaling with dimension
│   ├── FastICA_failure_modes.ipynb                 # [4] FastICA contrast failures
│   ├── OT-ICA_multi_algorithm_ablation.ipynb       # [5] 5-method × 5-regime benchmark
│   ├── OT-ICA_EEG_Artifact_application.ipynb       # [6] EEG blink artifact removal
│   ├── OT-ICA_price_discovery_application.ipynb    # [7] VECM price discovery
│   ├── OT-ICA_causal_discovery_application.ipynb   # [8] LiNGAM causal ordering
│   ├── TPM2026_workshop_paper_figures.ipynb         # [9] All paper figures (run last)
│   ├── w1_vs_w2_figure.py                          # Script: compact W1/W2 figure
│   ├── tpm2026_figures.py                          # Script: full ablation figure
│   ├── figs/                                       # Output PDFs
│   └── other/                                      # Exploratory / development notebooks
├── uai_workshop/              # TPM UAI 2026 workshop paper (LaTeX + compiled PDF)
├── report/                    # Master thesis (PDF + LaTeX)
├── slides/                    # Presentation slides
├── pyproject.toml
└── README.md
```

---

## Experiments — recommended order

Launch Jupyter from the `exp/` directory:

```bash
cd exp
jupyter lab
```

| # | Notebook | What it covers | Paper figure |
|---|----------|----------------|--------------|
| 1 | [OT-ICA_Methodology_Validation.ipynb](exp/OT-ICA_Methodology_Validation.ipynb) | End-to-end pipeline on a 4D mixture: whitening, W2² optimization, convergence, Amari error vs FastICA. **Run this first** to confirm the install works. | Fig. 1 (baseline convergence) |
| 2 | [OT-ICA_W1_vs_W2.ipynb](exp/OT-ICA_W1_vs_W2.ipynb) | Rotates a 2D mixture through 180° and plots the W1 vs W2² landscape and gradient profiles. Shows why W2² is a better ICA contrast. | App. Fig. (W1 vs W2) |
| 3 | [OT-ICA_stiefel_vs_FastICA_scaling.ipynb](exp/OT-ICA_stiefel_vs_FastICA_scaling.ipynb) | Amari error and wall-clock time vs dimension ($d = 5 \to 40$) for Laplace sources at $N=10{,}000$. Documents the curse-of-dimensionality ceiling shared by all empirical ICA methods. | App. Figs. (scaling) |
| 4 | [FastICA_failure_modes.ipynb](exp/FastICA_failure_modes.ipynb) | Constructs the zero-negentropy and vanishing-curvature distributions that analytically break FastICA; measures Amari error vs OT-ICA across dimensions. | App. Figs. (failure modes) |
| 5 | [OT-ICA_multi_algorithm_ablation.ipynb](exp/OT-ICA_multi_algorithm_ablation.ipynb) | 5-method × 5-mixture-regime × 3-dimension benchmark ($d \in \{10,20,30\}$, $N=10{,}000$, 10 trials). The main empirical comparison of the paper. | **Fig. 2 (main result)** |
| 6 | [OT-ICA_EEG_Artifact_application.ipynb](exp/OT-ICA_EEG_Artifact_application.ipynb) | Applies OT-ICA to frontal EEG (MNE sample dataset): isolates the ocular blink component by kurtosis, reconstructs the cleaned signal, reports RMS reduction. | Fig. 3 (EEG) |
| 7 | [OT-ICA_price_discovery_application.ipynb](exp/OT-ICA_price_discovery_application.ipynb) | Simulates a 3-market VECM, applies OT-ICA to recover structural shocks and Information Shares, compares to Cholesky. | Fig. 4 / Table 1 (VECM) |
| 8 | [OT-ICA_causal_discovery_application.ipynb](exp/OT-ICA_causal_discovery_application.ipynb) | Uses OT-ICA as a robust front-end to LiNGAM-style causal ordering on a linear SCM with heteroskedastic noise. | Thesis Ch. 6 |
| 9 | [TPM2026_workshop_paper_figures.ipynb](exp/TPM2026_workshop_paper_figures.ipynb) | Generates all final PDF figures for the workshop paper. Runs the full 750-trial ablation (compute-intensive; use a powerful machine). | All paper figures |

The scripts `exp/w1_vs_w2_figure.py` and `exp/tpm2026_figures.py` are standalone equivalents of notebooks 2 and 9 respectively, suitable for headless execution on a server.

---

## Reproducing paper figures

All figures in the UAI TPM 2026 workshop paper are produced by **notebook 9** (`TPM2026_workshop_paper_figures.ipynb`) or its script equivalent. The output PDFs are written to `exp/figs/` and copied to `uai_workshop/`. Running the full 750-trial ablation (3 dims × 5 configs × 5 methods × 10 trials) requires roughly 2–4 hours on a multi-core machine; set `ABL_TRIALS = 2` at the top of the script for a quick smoke test.

Thesis figures additionally use notebooks 3–8 and the exploratory notebooks under `exp/other/`.

---

## Key references

- Jha et al. (2026). *Linear ICA via Optimal Transport Metric as Contrast*. TPM @ UAI 2026. [`uai_workshop/tpm2026-ot_in_linear_ica.pdf`](uai_workshop/tpm2026-template.pdf)
- Jha, A. (2025). *Optimal Transport for Linear ICA* (Master's Thesis, University of Tübingen). [`report/Optimal_Transport_ICA_Master_Thesis_Ashutosh_Jha_v4.pdf`](report/Optimal_Transport_ICA_Master_Thesis_Ashutosh_Jha_v4.pdf)

---

## Contact

Ashutosh Jha — [ashutosh.jha@student.uni-tuebingen.de](mailto:ashutosh.jha@student.uni-tuebingen.de)

---

## License

MIT — see [LICENSE](LICENSE).
