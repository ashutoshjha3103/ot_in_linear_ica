"""
Shared constants/theme for the short_ppt animation scripts.

PALETTE below is copied verbatim from exp/tpm2026_figures.py (CELL 1) so the
slide animations use the exact same colorblind-safe per-method colors as the
paper's own figures. It is copied rather than imported because importing
exp/tpm2026_figures.py as a module executes its top-level experiment code
(750-trial ablation sweep + MNE EEG pipeline) — it is written to be pasted
into notebook cells, not imported.
"""
import matplotlib.pyplot as plt

PALETTE = {
    'OT-ICA':  '#D55E00',   # vermilion
    'FastICA': '#0173B2',   # blue
    'JADE':    '#029E73',   # teal
    'InfoMax': '#CC78BC',   # purple
    'Picard':  '#ECB22E',   # amber
}

FIG_BG = '#ffffff'
TEXT_COLOR = '#222222'


def set_deck_theme():
    plt.rcParams.update({
        'figure.facecolor': FIG_BG,
        'axes.facecolor': FIG_BG,
        'savefig.facecolor': FIG_BG,
        'text.color': TEXT_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'axes.edgecolor': TEXT_COLOR,
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.labelsize': 13,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.axisbelow': True,
    })


def save_animation(anim, out_path, fps=24):
    """Save as mp4 via ffmpeg; fall back to gif if ffmpeg is unavailable."""
    import shutil
    if shutil.which('ffmpeg') is not None and str(out_path).endswith('.mp4'):
        anim.save(out_path, writer='ffmpeg', fps=fps, dpi=130,
                   extra_args=['-pix_fmt', 'yuv420p'])
    else:
        gif_path = str(out_path).rsplit('.', 1)[0] + '.gif'
        anim.save(gif_path, writer='pillow', fps=fps, dpi=110)
        print(f'ffmpeg not found — saved {gif_path} instead')
