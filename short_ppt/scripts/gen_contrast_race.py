"""
contrast_race.mp4 -- the centerpiece animation. A whitened 2D toy mixture of
one super-Gaussian (Laplace) and one sub-Gaussian (Uniform) source, already
expressed in source-aligned coordinates (so the true source directions are
exactly theta=0 and theta=90 degrees). As a candidate direction b(theta)
sweeps the unit circle, four contrast functions are evaluated and plotted
side by side -- this is literally "what each method's contrast function
does": scan rotation angle, look for a non-Gaussianity maximum, and see
whether the peak lands on an independent (source-aligned) candidate.

Contrasts (all individually normalized to [0,1] to compare *where they peak*,
not their raw magnitude):
  - FastICA : squared logcosh-negentropy approximation
  - JADE    : squared excess kurtosis (4th-order cumulant)
  - InfoMax : |E[y tanh(y) - (1 - tanh(y)^2)]|, the tanh score-function
              contrast that drives InfoMax's natural gradient (and is the
              same A.5 curvature term discussed in the paper's appendix)
  - OT-ICA  : W2^2 to a standard Gaussian, via the project's own
              WassersteinICA (src/wasserstein_ica/core.py) for genuine
              numerical consistency with the paper.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(__file__))
from common import set_deck_theme, save_animation, PALETTE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
from wasserstein_ica import WassersteinICA

OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'media', 'contrast_race.mp4')

set_deck_theme()
rng = np.random.default_rng(2)
N = 12000

s1 = rng.laplace(0, 1 / np.sqrt(2), N)              # super-Gaussian, true axis 0 deg
s2 = rng.uniform(-np.sqrt(3), np.sqrt(3), N)        # sub-Gaussian,   true axis 90 deg

N_THETA = 160
thetas = np.linspace(0, np.pi, N_THETA, endpoint=False)
cos_t, sin_t = np.cos(thetas), np.sin(thetas)
Y = cos_t[:, None] * s1[None, :] + sin_t[:, None] * s2[None, :]   # (N_THETA, N)


def logcosh_stable(x):
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2 * ax)) - np.log(2)


# Gaussian reference constants (fixed large sample, reused for every theta)
g_ref = rng.normal(size=200_000)
E_logcosh_g = logcosh_stable(g_ref).mean()


def h_score(x):
    t = np.tanh(x)
    return x * t - 1 + t ** 2


E_h_g = h_score(g_ref).mean()

fastica = (logcosh_stable(Y).mean(axis=1) - E_logcosh_g) ** 2
m2 = (Y ** 2).mean(axis=1)
m4 = (Y ** 4).mean(axis=1)
jade = (m4 / m2 ** 2 - 3.0) ** 2
infomax = np.abs(h_score(Y).mean(axis=1) - E_h_g)

device = torch.device('cpu')
X_t = torch.tensor(np.stack([s1, s2]), dtype=torch.float32, device=device)
ica = WassersteinICA(X_t)
# s1, s2 are already independent & unit-variance by construction, so their
# covariance is ~I with near-degenerate eigenvalues. Calling .whiten() would
# run eigh() on that near-identity matrix, whose eigenvectors are numerically
# unstable when eigenvalues are nearly equal -- it silently picks an
# arbitrary rotation, which then misaligns theta=0/90 from the true axes.
# Bypass it and use the data as already-whitened.
ica.n = X_t.shape[1]
ica.X_white = X_t - X_t.mean(dim=1, keepdim=True)
ica.whitened = True
ica.analytical_target = ica._compute_analytical_target(ica.n)
B = torch.tensor(np.stack([cos_t, sin_t], axis=1), dtype=torch.float32, device=device)
with torch.no_grad():
    w2sq = ica.wasserstein2_analytical(B).cpu().numpy()


def normalize(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-12)


curves = {
    'FastICA': normalize(fastica),
    'JADE': normalize(jade),
    'InfoMax': normalize(infomax),
    'OT-ICA': normalize(w2sq),
}

# ---- figure ---------------------------------------------------------------
fig, (ax_scatter, ax_curve) = plt.subplots(1, 2, figsize=(11.5, 5.0))
fig.subplots_adjust(wspace=0.3, bottom=0.14, top=0.86, left=0.07, right=0.97)

R = 3.6
ax_scatter.scatter(s1, s2, s=5, color='#888888', alpha=0.35, zorder=1)
ax_scatter.set_xlim(-R, R)
ax_scatter.set_ylim(-R, R)
ax_scatter.set_aspect('equal')
ax_scatter.set_xticks([])
ax_scatter.set_yticks([])
ax_scatter.set_title('candidate direction $\\mathbf{b}(\\theta)$', fontsize=13)
proj_line, = ax_scatter.plot([], [], color=PALETTE['OT-ICA'], lw=3, zorder=3)
ax_scatter.annotate('true axis $s_1$ (Laplace)', xy=(R * 0.97, 0), xytext=(R * 0.5, R * 0.78),
                     fontsize=9, color='#666666',
                     arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1))
ax_scatter.annotate('true axis $s_2$ (Uniform)', xy=(0, R * 0.97), xytext=(-R * 0.95, R * 0.55),
                     fontsize=9, color='#666666',
                     arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1))

ax_curve.set_xlim(0, 180)
ax_curve.set_ylim(-0.05, 1.15)
ax_curve.set_xlabel(r'rotation angle $\theta$ (deg)')
ax_curve.set_ylabel('contrast (each curve normalized 0-1)')
ax_curve.axvline(0, color='#999999', lw=1, ls=':')
ax_curve.axvline(90, color='#999999', lw=1, ls=':')
ax_curve.set_title('what each contrast function sees', fontsize=13)

lines = {}
dots = {}
for name in ['FastICA', 'JADE', 'InfoMax', 'OT-ICA']:
    (lines[name],) = ax_curve.plot([], [], color=PALETTE[name], lw=2.5, label=name)
    (dots[name],) = ax_curve.plot([], [], 'o', color=PALETTE[name], ms=7)
ax_curve.legend(loc='upper center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.18))

deg = np.degrees(thetas)


def update(frame):
    theta = thetas[frame]
    proj_line.set_data([-R * np.cos(theta), R * np.cos(theta)],
                        [-R * np.sin(theta), R * np.sin(theta)])
    artists = [proj_line]
    for name in curves:
        lines[name].set_data(deg[:frame + 1], curves[name][:frame + 1])
        dots[name].set_data([deg[frame]], [curves[name][frame]])
        artists += [lines[name], dots[name]]
    return artists


anim = FuncAnimation(fig, update, frames=N_THETA, blit=False)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_animation(anim, OUT, fps=20)
print('wrote', OUT)
