"""
whitening.mp4 — raw correlated/off-center 2D mixture -> centered -> whitened
into an isotropic cloud -> unit circle of candidate directions b appears with
a rotating arrow, motivating why the ICA search space is the orthogonal
group once data is centered+whitened.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(__file__))
from common import set_deck_theme, save_animation, PALETTE

OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'media', 'whitening.mp4')

set_deck_theme()
rng = np.random.default_rng(1)
N = 6000

# Two i.i.d. Laplace sources, not bounded Uniform: a uniform (or any
# compact-support) source mixed linearly has a hard polytope edge whose
# geometry deterministically reveals the mixing matrix -- you could solve
# for it by fitting the minimal enclosing parallelogram, no statistics
# needed. That's a real but special-case identification trick (bounded-
# support/minimum-volume ICA), not the generic non-Gaussianity mechanism
# this talk is about. Two unbounded, smooth, equally-scaled Laplace sources
# give the classic "star/cross" shaped cloud instead: still visibly
# non-circular after whitening (so there's still a rotation to find), but
# from genuine higher-order structure, not a hard geometric boundary.
S = np.stack([
    rng.laplace(0, 1 / np.sqrt(2), N),
    rng.laplace(0, 1 / np.sqrt(2), N),
])

A = np.array([[2.0, 0.9], [0.4, 1.3]])
mean_shift = np.array([2.6, 1.6])
X_raw = A @ S + mean_shift[:, None]

mu = X_raw.mean(axis=1)
X_centered = X_raw - mu[:, None]

Cov = X_centered @ X_centered.T / N
evals, evecs = np.linalg.eigh(Cov)
W_white = (evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T)
X_white = W_white @ X_centered

# Percentile-based (not min/max) bounds: a handful of heavy-tailed Laplace
# outliers would otherwise blow up the view and shrink the bulk of the cloud
# to an imperceptible smudge. Bounds comfortably contain raw, centered,
# whitened cloud, and the unit-direction circle; a few extreme points may
# fall outside the visible window, which is fine for a schematic animation.
def p(arr, q):
    return np.percentile(arr, q)


lo_q, hi_q = 0.5, 99.5
xmin = min(p(X_raw[0], lo_q), p(X_centered[0], lo_q), -3.2) - 0.5
xmax = max(p(X_raw[0], hi_q), p(X_centered[0], hi_q), 3.2) + 0.5
ymin = min(p(X_raw[1], lo_q), p(X_centered[1], lo_q), -3.2) - 0.5
ymax = max(p(X_raw[1], hi_q), p(X_centered[1], hi_q), 3.2) + 0.5
half = max(xmax - xmin, ymax - ymin) / 2
cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
xmin, xmax = cx - half, cx + half
ymin, ymax = cy - half, cy + half

CIRCLE_R = 2.6  # schematic radius for the "unit circle of directions" overlay

# ---- frame schedule -------------------------------------------------------
F_HOLD_RAW   = 20
F_CENTER     = 55
F_HOLD_CEN   = 8
F_WHITEN     = 65
F_HOLD_WHITE = 15
F_CIRCLE_FADE = 18
F_ROTATE     = 100

schedule = (['hold_raw'] * F_HOLD_RAW + ['center'] * F_CENTER + ['hold_cen'] * F_HOLD_CEN +
            ['whiten'] * F_WHITEN + ['hold_white'] * F_HOLD_WHITE +
            ['circle'] * F_CIRCLE_FADE + ['rotate'] * F_ROTATE)
TOTAL = len(schedule)


def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep


fig, ax = plt.subplots(figsize=(6.4, 6.4))
fig.subplots_adjust(left=0.1, right=0.96, top=0.88, bottom=0.08)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
title = ax.set_title('', fontsize=15)
scat = ax.scatter(X_raw[0], X_raw[1], s=5, color=PALETTE['FastICA'], alpha=0.35, zorder=2)

theta_circle = np.linspace(0, 2 * np.pi, 200)
circle_line, = ax.plot(CIRCLE_R * np.cos(theta_circle), CIRCLE_R * np.sin(theta_circle),
                        color='#888888', lw=2, alpha=0.0, zorder=1)
label_b = ax.text(0, 0, '', fontsize=13, color=PALETTE['OT-ICA'], ha='center')
arrow_holder = {'artist': None}


def set_arrow(theta):
    """(Re)draw the radius arrow at angle theta, or hide it if theta is None.
    Annotation.set_alpha() does not reliably propagate to the arrow patch
    across matplotlib versions, so we remove/recreate instead of toggling alpha.
    """
    if arrow_holder['artist'] is not None:
        arrow_holder['artist'].remove()
        arrow_holder['artist'] = None
    if theta is not None:
        arrow_holder['artist'] = ax.annotate(
            '', xy=(CIRCLE_R * np.cos(theta), CIRCLE_R * np.sin(theta)), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color=PALETTE['OT-ICA'], lw=3), zorder=3)


def update(frame):
    stage = schedule[frame]
    stage_frames = [i for i, s in enumerate(schedule) if s == stage]
    local_t = (frame - stage_frames[0]) / max(1, (len(stage_frames) - 1))
    t = ease(np.clip(local_t, 0, 1))

    if stage == 'hold_raw':
        pts = X_raw
        title.set_text('Raw mixture $X = AS$')
        circle_line.set_alpha(0.0)
        set_arrow(None)
        label_b.set_text('')
    elif stage == 'center':
        pts = X_raw - t * mu[:, None]
        title.set_text(r'Centering: $X \leftarrow X - \mathbb{E}[X]$')
    elif stage == 'hold_cen':
        pts = X_centered
        title.set_text(r'Centered: $\mathbb{E}[X] = 0$')
    elif stage == 'whiten':
        M = (1 - t) * np.eye(2) + t * W_white
        pts = M @ X_centered
        title.set_text(r'Whitening: $\mathrm{Cov}(X) \to I$')
    elif stage == 'hold_white':
        pts = X_white
        title.set_text(r'Whitened: $\mathrm{Cov}(\widetilde{X}) = I$')
    elif stage == 'circle':
        pts = X_white
        circle_line.set_alpha(t)
        title.set_text('Search space: unit directions $\\mathbf{b}$')
    elif stage == 'rotate':
        pts = X_white
        circle_line.set_alpha(1.0)
        theta = 2 * np.pi * local_t
        set_arrow(theta)
        label_b.set_position((CIRCLE_R * 1.18 * np.cos(theta), CIRCLE_R * 1.18 * np.sin(theta)))
        label_b.set_text('$\\mathbf{b}$')
        title.set_text(r'Optimize over $\mathbf{b} \in$ orthogonal group $\mathcal{S}$')

    scat.set_offsets(pts.T)
    return [scat, circle_line, label_b, title]


anim = FuncAnimation(fig, update, frames=TOTAL, blit=False)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_animation(anim, OUT, fps=24)
print('wrote', OUT)
