# Optimal Transport for Linear Independent Component Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Master's Thesis Project**
**Author:** Ashutosh Jha  
**Program:** M.Sc. Quantitative Data Science Methods  
**Affiliations:** Eberhard Karls University of Tübingen & Empirical Inference Dept., Max Planck Institute for Intelligent Systems

---

## Abstract

This project investigates the application of **Optimal Transport (Wasserstein Distance)** for the task of **Independent Component Analysis (ICA)** in a linear mixture setting.

Standard ICA algorithms, such as FastICA, typically maximize statistical approximations of negentropy or kurtosis to identify non-Gaussian sources. This project implements a geometric approach: recovering independent sources by maximizing the **Squared Wasserstein-2 Distance ($W_2^2$)** between the projected data and a standard Normal distribution.

By leveraging **PyTorch Autograd** for Riemannian optimization on the unit sphere, this implementation provides a robust method for blind source seperation.

## Key Features

* **Metric:** Implements **Squared Wasserstein-2 ($W_2^2$)** distance using efficient 1D sorting and Hazen plotting positions for robust quantile estimation.
* **Optimization:** Utilizes **Projected Gradient Ascent** (via PyTorch Autograd) to maximize non-Gaussianity directly on the unit hypersphere.
Analysis (PCA) baselines.

## Repository Structure

```text
OT_IN_LINEAR_ICA/
├── src/
│   └── wasserstein_ica/   # Core package implementation
│       ├── __init__.py
│       └── core.py        # Main WassersteinICA class
├── examples/              # Jupyter notebooks and reproduction scripts
├── assets/                # Plots and diagrams generated for the thesis
├── pyproject.toml         # Build configuration
└── README.md
```

## Installation

To install the package in editable mode (allowing for code modifications and immediate testing):

```bash
git clone [https://github.com/ashutoshjha3103/OT_IN_LINEAR_ICA.git](https://github.com/ashutoshjha3103/OT_IN_LINEAR_ICA.git)
cd OT_IN_LINEAR_ICA
pip install -e .
```

## Quick Start

The following example demonstrates how to recover sources from a mixed signal using the `WassersteinICA` class.

```python
import torch
import numpy as np
from wasserstein_ica import WassersteinICA

# 1. Simulate Data (2 Sources, 1000 Samples)
# Using Laplace sources as an example of non-Gaussian data
S = np.random.laplace(0, 1, (2, 1000))
A = np.array([[0.8, 0.2], [0.3, 0.7]])  # Mixing Matrix
X = torch.tensor(np.dot(A, S), dtype=torch.float32)

# 2. Initialize and Whiten
# Whitening is a necessary preprocessing step to decorrelate the data
ica = WassersteinICA(X)
ica.whiten()

# 3. Optimize
# Finds the projection direction that maximizes the distance to Gaussianity
w_est, distance = ica.optimize_wasserstein2(continuous=True)

print(f"Recovered Source Direction: {w_est}")
print(f"Max Wasserstein Distance: {distance:.4f}")
```

## Methodology

### The Objective Function

We formulate ICA as a projection pursuit problem. The goal is to find a projection vector $w$ (where $\|w\|=1$) such that the projected data $Y = w^T X$ is maximally distant from a standard Gaussian distribution.

$$\underset{w \in \mathbb{R}^d, \|w\|=1}{\text{maximize}} \quad W_2^2(P_{w^TX}, \mathcal{N}(0, 1))$$

Here, $W_2^2$ denotes the Squared Wasserstein-2 distance, calculated empirically via the quantile function:

$$W_2^2 = \frac{1}{n} \sum_{i=1}^{n} (y_{(i)} - \Phi^{-1}(q_i))^2$$

* $y_{(i)}$: The sorted projected samples.
* $q_i$: Hazen plotting positions, defined as $\frac{i - 0.5}{n}$.
* $\Phi^{-1}$: The inverse CDF (quantile function) of the standard normal distribution.

### Optimization Landscape

Unlike $W_1$ (Mean Absolute Error), the Squared $W_2$ metric provides a smoother optimization landscape. This allows for more stable gradients, particularly when separating heavy-tailed distributions where standard moments might produce erratic optimization surfaces.

## Roadmap

- [x] Implement robust $W_2^2$ calculation.
- [x] Implement Autograd-based Projected Gradient Ascent.
- [x] Verify recovery on Laplace, Student-t, and Bernoulli sources.
- [ ] Extend to Multi-Source extraction (Deflationary or Symmetric approach).
- [ ] Compare performance benchmarks vs FastICA and InfoMax.
- [ ] Finalize Thesis manuscript.

## Contact

Ashutosh Jha ([ashutosh.jha@student.uni-tuebingen.de](mailto:ashutosh.jha@student.uni-tuebingen.de))

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.