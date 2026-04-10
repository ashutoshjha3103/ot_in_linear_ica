# Optimal Transport for Linear Independent Component Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Master's thesis project**  
**Author:** Ashutosh Jha  
**Program:** M.Sc. Quantitative Data Science Methods  
**Affiliations:** Eberhard Karls University of Tübingen & Empirical Inference Dept., Max Planck Institute for Intelligent Systems

---

## About this project

In **linear independent component analysis (ICA)**, the goal is to recover unknown non-Gaussian sources from observed mixtures. Classical algorithms such as FastICA typically rely on contrast functions or moment-based proxies for non-Gaussianity.

This repository implements a **Wasserstein / optimal-transport–based linear ICA** approach implemented in PyTorch: sources are sought by comparing projected data to a reference distribution in a transport-aware way, with optimization on the constraint set appropriate for ICA (e.g., normalized directions or orthogonal unmixing). The core library lives under `src/wasserstein_ica/`; `exp/` holds the main experimental notebooks that compare methods, validate behavior, and illustrate applications. The written thesis and LaTeX sources are under `report/`.

---

## Requirements

- **Python** 3.8 or newer  
- **pip** (recommended: recent `pip`, `setuptools`, and `wheel`)

Runtime dependencies are declared in [`pyproject.toml`](pyproject.toml). They include **NumPy**, **SciPy**, **PyTorch** (≥ 1.10), **POT** (Python Optimal Transport), **scikit-learn**, **pandas**, **matplotlib**, **seaborn**, **tqdm**, **joblib**, and **Jupyter** tooling (`jupyter`, `ipykernel`).

If you need a **GPU build of PyTorch** or a specific CUDA version, install PyTorch from the [official install matrix](https://pytorch.org/get-started/locally/) before or after the editable install (see below), so it matches your system.

---

## Set up a Python environment

From the repository root:

```bash
# Create and activate a virtual environment (example with venv)
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows (cmd/PowerShell)

python -m pip install --upgrade pip setuptools wheel
```

---

## Install `wasserstein-ica`

Clone the repository, enter it, then install in **editable** mode so changes under `src/` are picked up immediately:

```bash
git clone https://github.com/ashutoshjha3103/OT_IN_LINEAR_ICA.git
cd OT_IN_LINEAR_ICA
pip install -e .
```

This installs the package name **`wasserstein-ica`** (import as `wasserstein_ica`) and pulls in all dependencies from `pyproject.toml`.

Verify the install:

```bash
python -c "from wasserstein_ica import WassersteinICA; print('OK')"
```

---

## Run the experimental notebooks

Start Jupyter from an environment where `wasserstein-ica` is installed.

**Recommended:** launch Jupyter with the **`exp/`** directory as the working directory so notebook-relative paths (for example data files next to notebooks) resolve correctly:

```bash
cd exp
jupyter lab
# or: jupyter notebook
```

The table lists notebooks in the top level of `exp/` in a **recommended reading order** (core validation and comparisons first, then applications). Each row has two description lines, separated by a line break. Additional exploratory notebooks live under `exp/other/`.

| Notebook | What it does |
| --- | --- |
| [OT-ICA_Methodology_Validation.ipynb](exp/OT-ICA_Methodology_Validation.ipynb) | End-to-end sanity check of the OT-ICA pipeline in a manageable **4-dimensional** linear ICA instance, with plots and comparison to FastICA.<br>Run this first to confirm installs, whitening, and optimization behave as expected. |
| [OT-ICA_W1_vs_W2.ipynb](exp/OT-ICA_W1_vs_W2.ipynb) | Rotates a whitened 2D mixture through angles and compares **Wasserstein-1 vs Wasserstein-2** as ICA objectives via landscapes and numerical gradients.<br>Makes the case for W₂ here through smoother objectives and more reliable gradient structure for optimization. |
| [OT-ICA_stiefel_vs_FastICA_scaling.ipynb](exp/OT-ICA_stiefel_vs_FastICA_scaling.ipynb) | Benchmarks **scaling with dimension** for Stiefel-constrained OT-ICA versus FastICA on continuous Laplace mixtures, including accuracy (e.g. Amari error) and compute or wall-clock style summaries.<br>Documents how the transport-based approach trades off against the classical solver as problem size grows. |
| [FastICA_failure_modes.ipynb](exp/FastICA_failure_modes.ipynb) | Builds distributions that trigger **vanishing curvature** and related FastICA pitfalls (e.g. negentropy collapse), then measures unmixing quality (e.g. Amari error) versus OT-ICA across settings and dimensions.<br>Shows failures rooted in the contrast objective rather than iteration limits, and how OT-ICA behaves on the same data. |
| [OT-ICA_vs_FasICA_hybrid_and_discrete_dist_ablation_study.ipynb](exp/OT-ICA_vs_FasICA_hybrid_and_discrete_dist_ablation_study.ipynb) | **Experiment 1** stresses OT-ICA and FastICA on a high-dimensional mixture of many source types (heavy tails, bounded, skewed, and discrete marginals). **Experiment 2** targets discrete-source failure modes.<br>Together they test robustness when source statistics are heterogeneous rather than drawn from a single parametric family. |
| [causal_comp_analysis_application.ipynb](exp/causal_comp_analysis_application.ipynb) | Simulates a small linear SCM with location–scale (heteroskedastic) noise, mixes the latent variables orthogonally, and uses OT-ICA as a robust route to LiNGAM-style causal ordering where contrast-based ICA can fail.<br>Compares recovery of structure to illustrate when geometry-based separation helps causal component analysis. |
| [econometrics_application.ipynb](exp/econometrics_application.ipynb) | Applies OT-ICA to a simulated multi-market **price discovery** setting: observed prices are mixtures of latent shocks, and the notebook checks whether recovered components align with the underlying economic structure.<br>Use it as a stylized econometrics example beyond toy ICA benchmarks. |

If a notebook loads local files by name (e.g. tab-separated data in the same folder as the notebook), keep the kernel’s current working directory aligned with that notebook’s location.

---

## Repository layout

```text
OT_IN_LINEAR_ICA/
├── src/
│   └── wasserstein_ica/    # Installable package (WassersteinICA, core logic)
├── exp/                    # Main experiment notebooks (+ optional data next to notebooks)
│   └── other/              # Extra / exploratory notebooks
├── report/                 # Thesis text and LaTeX sources
├── pyproject.toml          # Package metadata and dependencies
├── LICENSE
└── README.md
```

---

## Quick start (code)

```python
import numpy as np
import torch
from wasserstein_ica import WassersteinICA

# Example: 2 sources, 1000 samples (Laplace sources)
rng = np.random.default_rng(0)
S = rng.laplace(0, 1, size=(2, 1000))
A = np.array([[0.8, 0.2], [0.3, 0.7]])
X = torch.tensor(A @ S, dtype=torch.float32)

ica = WassersteinICA(X)
ica.whiten()
w_est, distance = ica.optimize_wasserstein2(continuous=True)

print("Recovered direction:", w_est)
print("Wasserstein objective:", float(distance))
```

---

## Contact

Ashutosh Jha — [ashutosh.jha@student.uni-tuebingen.de](mailto:ashutosh.jha@student.uni-tuebingen.de)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
