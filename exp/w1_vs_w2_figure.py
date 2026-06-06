"""
Compact W1 vs W2 gradient-landscape figure for the UAI TPM 2026 workshop paper.
Saves to uai_workshop/w1_vs_w2_derivatives.pdf at full text width, 1×2 layout.

Usage:
    cd <repo-root>
    python exp/w1_vs_w2_figure.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from wasserstein_ica import WassersteinICA

# ── aesthetics ───────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})

COLOR_W2 = '#029E73'   # green
COLOR_W1 = '#CC78BC'   # purple

# ── data ─────────────────────────────────────────────────────────────────────
np.random.seed(42)
n_samples = 5000
S = np.random.standard_t(df=4, size=(2, n_samples))
A = np.array([[2.0, 0.0], [0.0, 0.5]])
X = torch.tensor(A @ S, dtype=torch.float32)

ica = WassersteinICA(X)
ica.whiten()

# ── sweep ─────────────────────────────────────────────────────────────────────
n_pts = 400
thetas = torch.linspace(0, np.pi, steps=n_pts)
ws = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)

w1_vals, w2_vals = [], []
for w in ws:
    w_n = w / torch.norm(w)
    w1_vals.append(ica.wasserstein1_distance(w_n).item())
    w2_vals.append(ica.wasserstein2_distance(w_n).item())

w1_vals  = np.array(w1_vals)
w2_vals  = np.array(w2_vals)
th_np    = thetas.numpy()

grad_w1 = np.gradient(w1_vals, th_np)
grad_w2 = np.gradient(w2_vals, th_np)
grad_w1_n = grad_w1 / np.max(np.abs(grad_w1))
grad_w2_n = grad_w2 / np.max(np.abs(grad_w2))

# ── plot ──────────────────────────────────────────────────────────────────────
TEXT_WIDTH_IN = 6.75   # full text width for one-column appendix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.38),
                                constrained_layout=True)

xticks      = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
xticklabels = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']

# panel 1 – landscape
ax1.plot(th_np, w2_vals, color=COLOR_W2, lw=1.6, label=r'$W_2^2$')
ax1.set_ylabel(r'$W_2^2$', color=COLOR_W2, labelpad=2)
ax1.tick_params(axis='y', labelcolor=COLOR_W2)
ax1_r = ax1.twinx()
ax1_r.plot(th_np, w1_vals, color=COLOR_W1, lw=1.6, label=r'$W_1$', linestyle='--')
ax1_r.set_ylabel(r'$W_1$', color=COLOR_W1, labelpad=2)
ax1_r.tick_params(axis='y', labelcolor=COLOR_W1)
ax1_r.spines['top'].set_visible(False)
ax1.set_xticks(xticks); ax1.set_xticklabels(xticklabels)
ax1.set_xlabel(r'Rotation angle $\theta$', labelpad=2)
ax1.set_title('(a) Objective landscape', loc='left', pad=3)
lines  = [mpl.lines.Line2D([0],[0], color=COLOR_W2, lw=1.6),
          mpl.lines.Line2D([0],[0], color=COLOR_W1, lw=1.6, linestyle='--')]
ax1.legend(lines, [r'$W_2^2$', r'$W_1$'], loc='lower center', ncol=2)

# panel 2 – gradients
ax2.axhline(0, color='gray', lw=1.0, alpha=0.5)
ax2.plot(th_np, grad_w2_n, color=COLOR_W2, lw=1.6, label=r'$\nabla_\theta W_2^2$')
ax2.plot(th_np, grad_w1_n, color=COLOR_W1, lw=1.6, linestyle='--', label=r'$\nabla_\theta W_1$')
ax2.set_ylabel('Normalized gradient', labelpad=2)
ax2.set_xlabel(r'Rotation angle $\theta$', labelpad=2)
ax2.set_xticks(xticks); ax2.set_xticklabels(xticklabels)
ax2.set_title('(b) Gradient profiles', loc='left', pad=3)
ax2.legend(loc='upper left', ncol=2)

# ── save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), '..', 'uai_workshop')
out_path = os.path.join(out_dir, 'w1_vs_w2_derivatives.pdf')
fig.savefig(out_path, format='pdf', bbox_inches='tight')
print(f"Saved → {os.path.abspath(out_path)}")
