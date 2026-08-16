"""
title_card.mp4 / closing_card.mp4 -- static title and closing cards for
the 3-5 min workshop video (uai_video/). Each is a still PNG held for a
few seconds via ffmpeg, at 1280x720 to match the video's target canvas.
"""
import os
import sys
import subprocess
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import set_deck_theme, FIG_BG, TEXT_COLOR

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'media')
os.makedirs(MEDIA_DIR, exist_ok=True)

set_deck_theme()

W, H, DPI = 1280, 720, 130


def make_card(png_path, draw_fn):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(FIG_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    draw_fn(ax)
    fig.savefig(png_path, facecolor=FIG_BG)
    plt.close(fig)


def draw_title(ax):
    ax.text(0.5, 0.62, 'Linear Independent Component Analysis\nvia Optimal Transport',
            ha='center', va='center', fontsize=22, fontweight='bold', color=TEXT_COLOR)
    ax.text(0.5, 0.44, 'Ashutosh Jha, Michel Besserve, Simon Buchholz',
            ha='center', va='center', fontsize=14, color=TEXT_COLOR)
    ax.text(0.5, 0.34, 'Tractable Probabilistic Modeling Workshop, UAI 2026',
            ha='center', va='center', fontsize=12, color='#666666')


def draw_closing(ax):
    ax.text(0.5, 0.60, 'Code & Paper', ha='center', va='center', fontsize=20,
            fontweight='bold', color=TEXT_COLOR)
    ax.text(0.5, 0.48, 'github.com/ashutoshjha3103/ot_in_linear_ica',
            ha='center', va='center', fontsize=13, color='#0173B2')
    ax.text(0.5, 0.32, 'Thank you', ha='center', va='center', fontsize=16, color=TEXT_COLOR)


def png_to_held_mp4(png_path, mp4_path, duration_s, fps=20):
    subprocess.run(['ffmpeg', '-y', '-loop', '1', '-i', png_path, '-t', str(duration_s),
                     '-r', str(fps), '-vf', f'scale={W}:{H}', '-pix_fmt', 'yuv420p', mp4_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


title_png = os.path.join(MEDIA_DIR, 'title_card.png')
closing_png = os.path.join(MEDIA_DIR, 'closing_card.png')
make_card(title_png, draw_title)
make_card(closing_png, draw_closing)

png_to_held_mp4(title_png, os.path.join(MEDIA_DIR, 'title_card.mp4'), duration_s=10)
png_to_held_mp4(closing_png, os.path.join(MEDIA_DIR, 'closing_card.mp4'), duration_s=15)
print('wrote title_card.mp4 and closing_card.mp4')
