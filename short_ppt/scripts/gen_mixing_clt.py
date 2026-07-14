"""
mixing_clt.mp4 — two non-Gaussian sources blend into a mixture that creeps
toward Gaussian as the mixing angle theta sweeps from a pure source (0)
through a balanced 50/50 mix (pi/4) to the other pure source (pi/2).
Right panel traces W2^2(b.Z, Gaussian) vs theta, foreshadowing the contrast
used throughout the rest of the deck.
"""
import os
import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(__file__))
from common import set_deck_theme, save_animation, PALETTE

OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'media', 'mixing_clt.mp4')

set_deck_theme()
rng = np.random.default_rng(0)
N = 6000

s1 = rng.laplace(0, 1 / np.sqrt(2), N)              # unit-variance Laplace
s2 = rng.uniform(-np.sqrt(3), np.sqrt(3), N)         # unit-variance Uniform

N_THETA = 140
thetas = np.linspace(0, np.pi / 2, N_THETA)


def w2sq_to_gaussian(y):
    y_sorted = np.sort(y)
    q = (np.arange(1, len(y) + 1) - 0.5) / len(y)
    target = scipy.stats.norm.ppf(q)
    return float(np.mean((y_sorted - target) ** 2))


w2_curve = np.array([w2sq_to_gaussian(np.cos(t) * s1 + np.sin(t) * s2) for t in thetas])

x_grid = np.linspace(-4, 4, 400)
gauss_pdf = scipy.stats.norm.pdf(x_grid)

fig, (ax_hist, ax_curve) = plt.subplots(1, 2, figsize=(10.5, 4.6))
fig.subplots_adjust(wspace=0.32, bottom=0.16, top=0.86, left=0.08, right=0.97)

bars = None
title = ax_hist.set_title('')
ax_hist.set_xlim(-4, 4)
ax_hist.set_ylim(0, 0.62)
ax_hist.set_xlabel(r'$y(\boldsymbol{\theta}) = s_1\cos\boldsymbol{\theta} + s_2\sin\boldsymbol{\theta}$')
ax_hist.set_ylabel('density')
gauss_line, = ax_hist.plot(x_grid, gauss_pdf, color='#999999', lw=2, ls='--', label='target $\\mathcal{N}(0,1)$')
ax_hist.legend(loc='upper right', frameon=False, fontsize=10)

ax_curve.plot(np.degrees(thetas), w2_curve, color='#cccccc', lw=6, solid_capstyle='round', zorder=1)
trace_line, = ax_curve.plot([], [], color=PALETTE['OT-ICA'], lw=3, zorder=2)
dot, = ax_curve.plot([], [], 'o', color=PALETTE['OT-ICA'], ms=9, zorder=3)
ax_curve.set_xlim(0, 90)
ax_curve.set_ylim(0, w2_curve.max() * 1.15)
ax_curve.set_xlabel(r'mixing angle $\theta$ (deg)')
ax_curve.set_ylabel(r'$W_2^2(y(\theta),\,\mathcal{N})$')
ax_curve.set_title('non-Gaussianity of the mixture', fontsize=13)
ax_curve.axvline(0, color='#888888', lw=1, ls=':')
ax_curve.axvline(90, color='#888888', lw=1, ls=':')
ax_curve.text(0, w2_curve.max() * 1.2, 'pure $s_1$', ha='center', fontsize=9, color='#888888')
ax_curve.text(90, w2_curve.max() * 1.2, 'pure $s_2$', ha='center', fontsize=9, color='#888888')

# ping-pong the sweep so the loop has no visual jump
frame_idx = list(range(N_THETA)) + list(range(N_THETA - 1, -1, -1))


def update(frame):
    global bars
    i = frame_idx[frame]
    theta = thetas[i]
    y = np.cos(theta) * s1 + np.sin(theta) * s2
    if bars is not None:
        for b in bars:
            b.remove()
    counts, edges = np.histogram(y, bins=60, range=(-4, 4), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bars = ax_hist.bar(centers, counts, width=edges[1] - edges[0],
                        color=PALETTE['FastICA'], alpha=0.75, zorder=2)
    title.set_text(f'$\\theta = {np.degrees(theta):4.0f}^\\circ$   '
                    f'$W_2^2$ to Gaussian $= {w2_curve[i]:.3f}$')
    trace_line.set_data(np.degrees(thetas[:i + 1]), w2_curve[:i + 1])
    dot.set_data([np.degrees(theta)], [w2_curve[i]])
    return list(bars) + [trace_line, dot, title]


anim = FuncAnimation(fig, update, frames=len(frame_idx), blit=False)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_animation(anim, OUT, fps=24)
print('wrote', OUT)
