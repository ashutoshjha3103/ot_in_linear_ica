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

Open notebooks at the top level of `exp/` for the main experiments (methodology validation, comparisons with FastICA, scaling, applications, etc.). Additional exploratory notebooks are under `exp/other/`.

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
