"""
mixer_unmixer_intro.mp4 -- schematic explainer for the ICA problem itself,
meant to run right after the title card. Staged like a short acting scene,
timed for narration to talk over each entrance in turn:

  1. Wall (with a gap) present from frame 0.
  2. Mixer pops in alone.
  3. ~1.5s later, Unmixer pops in -- and immediately settles into an idle
     "not looking, arms folded" stance (nothing to look at yet).
  4. The mixing system appears: sources S1/S2 and a gear labelled "A"
     (the mixing matrix), spinning clockwise. Gear "A" stays visible and
     turning for a couple of seconds so the viewer clearly sees it.
  5. The Mixer glances toward the Unmixer, then drops a tarp over gear
     "A" -- from then on it's hidden (still turning underneath, just not
     shown). Sources keep flowing into the hidden gear.
  6. A traveling mixed-signal wave (not a plain arrow) emerges from
     behind the tarp and starts moving toward the wall gap. This is the
     trigger: only now does the unmixing system -- a gear labelled "ICA",
     spinning counter-clockwise -- first appear, and only now does the
     Unmixer's pose transition from idle to attentive (unfolding, turning
     to look), timed to the wave's arrival.
  7. Two recovered outputs grow out of the ICA gear -- same histogram
     shapes as Z1/Z2, but swapped in order and rescaled (permutation +
     scale ambiguity).

Sources/outputs are drawn as small histograms of real samples (Laplace
for Z1, Uniform for Z2), not waveforms -- this is deliberately not a
"signal" in the signals-and-systems sense (no time axis, no continuous
trace). It's a batch of i.i.d. draws from a distribution, matching what
the method actually operates on. The traveling "X = AZ" object is a real
histogram of an actual linear mix of the same two sample sets.

Honest scoping note: stick figures can't carry real facial expression --
the "glance" / "idle vs. attentive" beats are crude approximations
(head/eye offsets, arm-pose blends, an opaque tarp patch). They read as
stylized, not genuinely emotive; narration should state "the Unmixer
never sees S or A" explicitly rather than relying on the animation alone.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon, Rectangle, FancyArrowPatch

sys.path.insert(0, os.path.dirname(__file__))
from common import set_deck_theme, save_animation, PALETTE

OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'media', 'mixer_unmixer_intro.mp4')

set_deck_theme()

S1_COLOR = PALETTE['FastICA']   # blue -- smooth "sine-like" source
S2_COLOR = PALETTE['JADE']      # teal -- spiky "square-like" source
X_COLOR = '#555555'
GEAR_FACE = '#e8e8e8'
GEAR_EDGE = '#333333'
TARP_COLOR = '#8a8a8a'
FIGURE_COLOR = '#333333'

FPS = 20

# ---- staged timeline (frames) -------------------------------------------
F_MIXER_IN = (5, 20)
F_UNMIXER_IN = (35, 50)        # ~1.5s after the mixer starts appearing
F_MIXING_SYS_IN = (55, 75)     # sources + gear A pop in together
REVEAL_HOLD_END = 115          # gear A sits visible & turning, uncovered, until here
F_GLANCE = (115, 135)
F_TARP = (135, 170)
F_FLOW_IN = (170, 215)
F_WAVE = (215, 260)            # wave travels from gear A to gear ICA
F_ICA_IN = (215, 230)          # gear ICA first appears -- triggered by the wave starting
F_LOOK_AT = (215, 245)         # unmixer: idle -> attentive, timed to the wave's arrival
F_OUTPUTS = (260, 295)
F_HOLD = (295, 415)            # extra-long hold: narration points out the scale/permutation ambiguity here
TOTAL = 415


def prog(f, span):
    a, b = span
    return float(np.clip((f - a) / (b - a), 0.0, 1.0))


def ease(t):
    return t * t * (3 - 2 * t)


# Sources/outputs are small histograms of real samples, not waveforms --
# see module docstring for why. Precomputed once at import time.
_rng_hist = np.random.default_rng(7)
_N_HIST_SAMPLES = 6000
_HIST_RANGE = (-3.2, 3.2)
_N_BINS = 9

_laplace_samples = _rng_hist.laplace(0, 1 / np.sqrt(2), _N_HIST_SAMPLES)
_uniform_samples = _rng_hist.uniform(-np.sqrt(3), np.sqrt(3), _N_HIST_SAMPLES)
_mixed_samples = 0.7 * _laplace_samples + 0.7 * _uniform_samples  # a genuine mixed batch


def _hist(samples):
    counts, edges = np.histogram(samples, bins=_N_BINS, range=_HIST_RANGE)
    heights = counts / counts.max()
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, heights


Z1_HIST = _hist(_laplace_samples)   # peaked, super-Gaussian
Z2_HIST = _hist(_uniform_samples)   # flat, sub-Gaussian
X_HIST = _hist(_mixed_samples)      # blended -- visibly more bell-shaped (CLT)

_BAR_MAX_H = 0.11


def draw_mini_hist(ax, cx, cy, hist, color, t, w=0.16, h_scale=1.0, amp=1.0, label=None, zorder=5, alpha=1.0):
    if alpha <= 0:
        return
    centers, heights = hist
    lo, hi = _HIST_RANGE
    xs = cx + (centers - (lo + hi) / 2) / (hi - lo) * w
    bar_w = (w / len(centers)) * 0.82
    for i, (x, hgt) in enumerate(zip(xs, heights)):
        jitter = 1 + 0.05 * np.sin(2 * np.pi * (0.35 * t + i * 0.41))  # gentle "live sampling" breathing
        bh = max(hgt, 0.03) * jitter * h_scale * amp * _BAR_MAX_H
        ax.add_patch(Rectangle((x - bar_w / 2, cy), bar_w, bh, facecolor=color, edgecolor='none',
                                alpha=alpha, zorder=zorder))
    if label:
        top = cy + h_scale * amp * _BAR_MAX_H * 1.08 + 0.035
        ax.text(cx, top, label, ha='center', va='bottom', fontsize=9, color=color, zorder=zorder, alpha=alpha)


def gear_polygon(cx, cy, r_outer, r_inner, n_teeth, angle_deg):
    pts = []
    for i in range(n_teeth):
        a0 = 2 * np.pi * i / n_teeth
        a1 = 2 * np.pi * (i + 0.28) / n_teeth
        a2 = 2 * np.pi * (i + 0.5) / n_teeth
        a3 = 2 * np.pi * (i + 0.78) / n_teeth
        for r, a in ((r_inner, a0), (r_outer, a1), (r_outer, a2), (r_inner, a3)):
            pts.append((r, a))
    offset = np.radians(angle_deg)
    return [(cx + r * np.cos(a + offset), cy + r * np.sin(a + offset)) for r, a in pts]


def draw_gear(ax, cx, cy, angle_deg, r_outer, r_inner, n_teeth, letter, zorder=4, alpha=1.0):
    if alpha <= 0:
        return
    poly = Polygon(gear_polygon(cx, cy, r_outer, r_inner, n_teeth, angle_deg),
                    closed=True, facecolor=GEAR_FACE, edgecolor=GEAR_EDGE, lw=1.3, zorder=zorder, alpha=alpha)
    ax.add_patch(poly)
    hub = Circle((cx, cy), r_inner * 0.55, facecolor='#ffffff', edgecolor=GEAR_EDGE, lw=1.0, zorder=zorder + 1, alpha=alpha)
    ax.add_patch(hub)
    ax.text(cx, cy, letter, ha='center', va='center', fontsize=13, fontweight='bold',
            color='#222222', zorder=zorder + 2, alpha=alpha)


def draw_stick_figure(ax, cx, cy, color, label, arm_fold=0.0, eye_dx=0.0, eye_dy=0.0, head_dx=0.0, alpha=1.0):
    if alpha <= 0:
        return
    hx, hy = cx + head_dx, cy + 0.10
    head = Circle((hx, hy), 0.045, facecolor=color, edgecolor=FIGURE_COLOR, lw=1.2, zorder=5, alpha=alpha)
    ax.add_patch(head)
    eye = Circle((hx + eye_dx, hy + eye_dy), 0.010, facecolor='#222222', edgecolor='none', zorder=6, alpha=alpha)
    ax.add_patch(eye)
    ax.plot([cx, cx], [cy + 0.055, cy - 0.08], color=color, lw=2.5, zorder=5, alpha=alpha)
    arm_l_normal = ((cx, cx - 0.06), (cy + 0.02, cy - 0.03))
    arm_r_normal = ((cx, cx + 0.06), (cy + 0.02, cy - 0.03))
    arm_l_folded = ((cx, cx + 0.045), (cy - 0.01, cy - 0.015))
    arm_r_folded = ((cx, cx - 0.045), (cy - 0.02, cy - 0.015))
    for normal, folded in ((arm_l_normal, arm_l_folded), (arm_r_normal, arm_r_folded)):
        xs = tuple(n + (f - n) * arm_fold for n, f in zip(normal[0], folded[0]))
        ys = tuple(n + (f - n) * arm_fold for n, f in zip(normal[1], folded[1]))
        ax.plot(xs, ys, color=color, lw=2, zorder=5, alpha=alpha)
    ax.plot([cx, cx - 0.05], [cy - 0.08, cy - 0.18], color=color, lw=2, zorder=5, alpha=alpha)
    ax.plot([cx, cx + 0.05], [cy - 0.08, cy - 0.18], color=color, lw=2, zorder=5, alpha=alpha)
    ax.text(cx, cy - 0.27, label, ha='center', va='center', fontsize=12, fontweight='bold',
            color=FIGURE_COLOR, alpha=alpha)


def growing_arrow(ax, start, end, progress, color=X_COLOR, lw=2.0, zorder=3):
    if progress <= 0:
        return
    sx, sy = start
    ex, ey = end
    cx, cy = sx + (ex - sx) * progress, sy + (ey - sy) * progress
    arrow = FancyArrowPatch((sx, sy), (cx, cy), arrowstyle='-|>', mutation_scale=13,
                             color=color, lw=lw, zorder=zorder)
    ax.add_patch(arrow)


# ---- fixed layout --------------------------------------------------------
MIXER_XY = (-0.62, 0.80)
UNMIXER_XY = (0.62, 0.80)
S1_XY = (-0.90, 0.44)
S2_XY = (-0.90, 0.18)
GEAR_A_XY = (-0.55, 0.30)
GEAR_ICA_XY = (0.55, 0.30)
GAP_Y = 0.30
R_OUT, R_IN, N_TEETH = 0.115, 0.085, 10
OUT1_XY = (0.90, 0.44)
OUT2_XY = (0.90, 0.16)


def draw_frame(ax, frame):
    ax.clear()
    ax.set_xlim(-1.15, 1.05)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.plot([0, 0], [0.62, -0.15], color=FIGURE_COLOR, lw=2.5, zorder=2)
    ax.plot([0, 0], [GAP_Y + 0.06, 1.05], color=FIGURE_COLOR, lw=2.5, zorder=2)

    t_glob = frame / FPS

    mixer_alpha = ease(prog(frame, F_MIXER_IN))
    unmixer_alpha = ease(prog(frame, F_UNMIXER_IN))
    sys_alpha = ease(prog(frame, F_MIXING_SYS_IN))

    if mixer_alpha > 0:
        glance = 1 - abs(2 * prog(frame, F_GLANCE) - 1) if F_GLANCE[0] <= frame <= F_GLANCE[1] else 0.0
        glance = max(0.0, glance)
        draw_stick_figure(ax, *MIXER_XY, S1_COLOR, 'Mixer', head_dx=0.02 * glance, eye_dx=0.018 * glance,
                           alpha=mixer_alpha)

    if unmixer_alpha > 0:
        attentive = ease(prog(frame, F_LOOK_AT))
        fold = 1 - attentive
        eye_dx = 0.02 * (1 - attentive) + (-0.015) * attentive
        eye_dy = 0.02 * (1 - attentive) + (-0.025) * attentive
        draw_stick_figure(ax, *UNMIXER_XY, S2_COLOR, 'Unmixer', arm_fold=fold, eye_dx=eye_dx, eye_dy=eye_dy,
                           alpha=unmixer_alpha)

    if sys_alpha > 0:
        draw_mini_hist(ax, *S1_XY, Z1_HIST, S1_COLOR, t_glob, label='$Z_1$', alpha=sys_alpha)
        draw_mini_hist(ax, *S2_XY, Z2_HIST, S2_COLOR, t_glob, label='$Z_2$', alpha=sys_alpha)

        angle_a = -frame * 5.0
        draw_gear(ax, *GEAR_A_XY, angle_a, R_OUT, R_IN, N_TEETH, 'A', alpha=sys_alpha)
        ax.text(GEAR_A_XY[0], GEAR_A_XY[1] - R_OUT - 0.06, 'MIXING', ha='center', va='center',
                fontsize=10, fontweight='bold', color=FIGURE_COLOR, zorder=7, alpha=sys_alpha)

        tarp_progress = ease(prog(frame, F_TARP))
        if tarp_progress > 0:
            pad = 0.03
            x0 = GEAR_A_XY[0] - R_OUT - pad
            x1 = GEAR_A_XY[0] + R_OUT + pad
            top = GEAR_A_XY[1] + R_OUT + pad
            full_h = 2 * (R_OUT + pad)
            cur_h = full_h * tarp_progress
            tarp = Rectangle((x0, top - cur_h), x1 - x0, cur_h,
                              facecolor=TARP_COLOR, edgecolor='#555555', lw=1.0, alpha=0.9 * sys_alpha, zorder=6)
            ax.add_patch(tarp)
            if tarp_progress > 0.98:
                for fy in (0.35, 0.55, 0.75):
                    ax.plot([x0 + 0.01, x1 - 0.01], [top - full_h * fy] * 2, color='#777777', lw=0.8,
                            alpha=0.6 * sys_alpha, zorder=7)

        p_flow = ease(prog(frame, F_FLOW_IN))
        growing_arrow(ax, (S1_XY[0] + 0.10, S1_XY[1]), (GEAR_A_XY[0] - R_OUT - 0.03, GEAR_A_XY[1] + 0.03), p_flow, S1_COLOR)
        growing_arrow(ax, (S2_XY[0] + 0.10, S2_XY[1]), (GEAR_A_XY[0] - R_OUT - 0.03, GEAR_A_XY[1] - 0.03), p_flow, S2_COLOR)

        p_wave = ease(prog(frame, F_WAVE))
        if p_wave > 0:
            wave_start_x = GEAR_A_XY[0] + R_OUT
            wave_end_x = GEAR_ICA_XY[0] - R_OUT - 0.02
            wave_cx = wave_start_x + (wave_end_x - wave_start_x) * p_wave
            draw_mini_hist(ax, wave_cx, GAP_Y, X_HIST, X_COLOR, t_glob, w=0.14, h_scale=0.8, zorder=3,
                            label='$X = AZ$')

    ica_alpha = ease(prog(frame, F_ICA_IN))
    if ica_alpha > 0:
        angle_ica = frame * 6.0
        draw_gear(ax, *GEAR_ICA_XY, angle_ica, R_OUT, R_IN, N_TEETH, 'ICA', alpha=ica_alpha)
        ax.text(GEAR_ICA_XY[0], GEAR_ICA_XY[1] - R_OUT - 0.06, 'UNMIXING', ha='center', va='center',
                fontsize=10, fontweight='bold', color=FIGURE_COLOR, zorder=7, alpha=ica_alpha)

    p_out = ease(prog(frame, F_OUTPUTS))
    if p_out > 0:
        growing_arrow(ax, (GEAR_ICA_XY[0] + R_OUT + 0.02, GEAR_ICA_XY[1] + 0.03), (OUT1_XY[0] - 0.10, OUT1_XY[1]), p_out, S2_COLOR)
        growing_arrow(ax, (GEAR_ICA_XY[0] + R_OUT + 0.02, GEAR_ICA_XY[1] - 0.03), (OUT2_XY[0] - 0.10, OUT2_XY[1]), p_out, S1_COLOR)
    if p_out >= 1.0:
        # recovered outputs: same histogram shapes as Z1/Z2, swapped order + rescaled
        draw_mini_hist(ax, *OUT1_XY, Z2_HIST, S2_COLOR, t_glob, amp=1.6, label=r'$\hat{Z}_1$')
        draw_mini_hist(ax, *OUT2_XY, Z1_HIST, S1_COLOR, t_glob, amp=0.6, label=r'$\hat{Z}_2$')

    return []


fig, ax = plt.subplots(figsize=(11.5, 6.2))
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02)
fig.suptitle('Linear ICA: Recovering true sources from only a mixture',
             fontsize=15, fontweight='bold', y=0.97)


def update(frame):
    return draw_frame(ax, frame)


anim = FuncAnimation(fig, update, frames=TOTAL, blit=False)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_animation(anim, OUT, fps=FPS)
print('wrote', OUT)
