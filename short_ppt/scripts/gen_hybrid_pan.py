"""
hybrid_test_pan.mp4 -- Ken-Burns pan/zoom over the paper's actual headline
figure (general_hybrid_test.pdf -> .png), not a script-generated plot, so
it can't get a matplotlib fig.suptitle like the other clips. Instead: pad
the figure onto a 16:9 canvas with generous margin, draw the title into
the top margin with PIL using matplotlib's own DejaVu Sans Bold for visual
consistency with the other clips' titles, then a slow zoom via ffmpeg
zoompan.

The zoom is TOP-ANCHORED (fixed y=0), not center-anchored. A center-anchor
crops symmetrically from the canvas center outward, which eats into the
title almost immediately since it sits near the top edge, far from center
-- no zoom speed fixes that, it needs a different anchor. Top-anchoring
keeps the title band fixed in frame while the zoom still visibly tightens
on the chart below it. MAX_ZOOM is computed from the actual title+figure
extents so the crop never crops content at any point in the zoom (same
"leave a safety margin" lesson as the earlier center-zoom version, which
first shipped without enough padding and cropped row labels mid-zoom).
"""
import os
import subprocess
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, '..', 'assets', 'figures', 'general_hybrid_test.png')
MEDIA = os.path.join(HERE, '..', 'assets', 'media')
PADDED = os.path.join(MEDIA, 'hybrid_padded.png')
OUT = os.path.join(MEDIA, 'hybrid_test_pan.mp4')

TITLE = 'ICA Methods performance at higher dimensions with varied source distributions'
CANVAS_W, CANVAS_H = 4512, 2538  # 16:9
DURATION_S = 50
SAFETY = 0.97  # shrink the computed max zoom by this factor for a buffer

font_path = next(f.fname for f in fm.fontManager.ttflist if 'DejaVuSans-Bold.ttf' in f.fname)

src = Image.open(SRC).convert('RGB')
canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), 'white')
ox, oy = (CANVAS_W - src.width) // 2, (CANVAS_H - src.height) // 2
canvas.paste(src, (ox, oy))

draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype(font_path, 68)
bbox = draw.textbbox((0, 0), TITLE, font=font)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
margin_h = oy  # top margin band above the pasted figure
title_y = (margin_h - th) / 2 - bbox[1]
draw.text(((CANVAS_W - tw) / 2, title_y), TITLE, font=font, fill='#111111')

canvas.save(PADDED)

# Content that must stay in frame at every zoom level: from a small buffer
# above the title down to the bottom of the pasted figure (its x-axis labels).
content_bottom = oy + src.height
max_zoom = SAFETY * CANVAS_H / content_bottom
zoom_rate = (max_zoom - 1) / (DURATION_S * 20)

subprocess.run([
    'ffmpeg', '-y', '-loop', '1', '-framerate', '20', '-i', PADDED,
    '-vf', f"zoompan=z='min(1+{zoom_rate}*on,{max_zoom})':"
           f"x='iw/2-(iw/zoom/2)':y='0':d=1:s=1280x720:fps=20",
    '-t', str(DURATION_S), '-pix_fmt', 'yuv420p', OUT, '-loglevel', 'error',
], check=True)
print(f'wrote {OUT}  (max_zoom={max_zoom:.3f})')
