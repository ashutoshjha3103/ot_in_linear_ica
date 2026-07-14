"""
cardoso.png -- static schematic of Cardoso's Pythagorean identity: two right
triangles sharing a hypotenuse [P_Y -> N(diag Cov Y)], one through the
Product manifold P (closest independent distribution), one through the
Gaussian manifold G (closest Gaussian). Shows the two underlying KL
Pythagorean decompositions (one per triangle), equates them (shared
hypotenuse), and gives the resulting unified identity plus its
post-whitening simplification I(Y) = G(Y) - sum_i G(Y_i).

Uses \\mathcal{} throughout (not \\mathscr{}) to match the notation rendered
by KaTeX elsewhere in the deck -- matplotlib's mathtext renders \\mathscr
with a different, inconsistent-looking script glyph.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import set_deck_theme, PALETTE

OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'figures', 'cardoso.png')
set_deck_theme()

# ---- schematic geometry ----------------------------------------------------
P_Y = (0.0, 4.0)
P_DIAG = (0.0, 0.0)
center, radius = (0.0, 2.0), 2.0
P_PROD = (radius * np.cos(np.radians(160)), center[1] + radius * np.sin(np.radians(160)))
P_GAUSS = (radius * np.cos(np.radians(20)), center[1] + radius * np.sin(np.radians(20)))

fig, ax = plt.subplots(figsize=(10.5, 7.5))
fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03)
ax.set_xlim(-6.8, 6.8)
ax.set_ylim(-4.7, 5.0)
ax.set_aspect('equal')
ax.axis('off')

hyp_line, = ax.plot([P_Y[0], P_DIAG[0]], [P_Y[1], P_DIAG[1]], color='#bbbbbb', lw=1.5, ls=':', zorder=1)

ax.plot([P_Y[0], P_PROD[0]], [P_Y[1], P_PROD[1]], color=PALETTE['FastICA'], lw=3.2, zorder=2)
ax.plot([P_PROD[0], P_DIAG[0]], [P_PROD[1], P_DIAG[1]], color=PALETTE['FastICA'], lw=3.2, zorder=2)
ax.plot([P_Y[0], P_GAUSS[0]], [P_Y[1], P_GAUSS[1]], color=PALETTE['OT-ICA'], lw=3.2, zorder=2)
ax.plot([P_GAUSS[0], P_DIAG[0]], [P_GAUSS[1], P_DIAG[1]], color=PALETTE['OT-ICA'], lw=3.2, zorder=2)

mid = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
lerp = lambda a, b, t: (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
# I(Y) and G(Y) sit on the two legs nearest the apex, where the legs are
# close together -- anchor them further down the leg (away from the apex)
# and push outward so the larger font doesn't collide at the top.
ix, iy = lerp(P_Y, P_PROD, 0.78)
gx, gy = lerp(P_Y, P_GAUSS, 0.78)
ax.text(ix, iy + 0.45, '$\\mathcal{I}(Y)$', fontsize=22.5, color=PALETTE['FastICA'], ha='center', va='bottom', zorder=4)
ax.text(*mid(P_PROD, P_DIAG), '$\\sum_i G(Y_i)$  ', fontsize=22.5, color=PALETTE['FastICA'], ha='right', va='top', zorder=4)
ax.text(gx, gy + 0.45, '$G(Y)$', fontsize=22.5, color=PALETTE['OT-ICA'], ha='center', va='bottom', zorder=4)
ax.text(*mid(P_GAUSS, P_DIAG), '  $C(Y)$', fontsize=22.5, color=PALETTE['OT-ICA'], ha='left', va='top', zorder=4)

# points + labels, each naming both the distribution and the manifold it sits on
pt_specs = [
    (P_Y, '$P_Y$', '(joint distribution)', (0, 0.4), 'bottom', 'center', '#222222'),
    (P_DIAG, '$\\mathcal{N}(\\mathrm{diag\\,Cov}\\,Y)$', '', (0, -0.55), 'top', 'center', '#222222'),
    (P_PROD, '$P_Y^P = \\prod_i P_{Y_i}$', 'closest independent,\nProduct manifold $\\mathcal{P}$',
     (-0.4, 0.3), 'bottom', 'right', PALETTE['FastICA']),
    (P_GAUSS, '$\\mathcal{N}(\\mathrm{Cov}\\,Y)$', 'closest Gaussian,\nGaussian manifold $\\mathcal{G}$',
     (0.4, 0.3), 'bottom', 'left', PALETTE['OT-ICA']),
]
for (x, y), main, sub, (dx, dy), va, ha, color in pt_specs:
    ax.plot([x], [y], 'o', color='#222222', ms=13, zorder=5)
    label = main if not sub else f'{main}\n{sub}'
    ax.text(x + dx, y + dy, label, fontsize=20.7, ha=ha, va=va, color=color, zorder=5)

NDIAG = r'\mathcal{N}(\mathrm{diag\,Cov}\,Y)'

# The two Pythagorean (KL-divergence) decompositions, one per triangle, each
# stating hypotenuse = leg1 + leg2 -- this is the actual theorem being drawn,
# not just labels on the legs.
ax.text(0, -1.35, rf'$D_{{\mathrm{{KL}}}}\left(P_Y \,\|\, {NDIAG}\right) \;=\; \mathcal{{I}}(Y) + \sum_i G(Y_i)$',
        fontsize=20.7, ha='center', va='center', color=PALETTE['FastICA'])
ax.text(0, -2.15, rf'$D_{{\mathrm{{KL}}}}\left(P_Y \,\|\, {NDIAG}\right) \;=\; G(Y) + C(Y)$',
        fontsize=20.7, ha='center', va='center', color=PALETTE['OT-ICA'])

ax.plot([-1.8, 1.8], [-2.7, -2.7], color='#999999', lw=1.3)
ax.text(1.95, -2.7, 'same LHS', fontsize=14.4, color='#888888', ha='left', va='center')

ax.text(0, -3.4, r'$\mathcal{I}(Y) + \sum_i G(Y_i) \;=\; G(Y) + C(Y)$',
        fontsize=22.5, ha='center', va='center', color='#222222')
ax.text(0, -4.15, r'whitened ($C(Y){=}0$):   $\mathcal{I}(Y) = G(Y) - \sum_i G(Y_i)$',
        fontsize=22.5, ha='center', va='center', color=PALETTE['OT-ICA'], fontweight='bold')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches='tight', pad_inches=0.15)
print('wrote', OUT)
