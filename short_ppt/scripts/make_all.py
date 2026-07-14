"""Regenerate every animation used in the deck. Run from anywhere with:
    python3 short_ppt/scripts/make_all.py
"""
import runpy
import os

SCRIPTS = [
    'gen_mixing_clt.py',
    'gen_centering_whitening.py',
    'gen_contrast_race.py',
    'gen_cardoso_pythagoras.py',
    'gen_eeg_scroll.py',
]

HERE = os.path.dirname(__file__)

for script in SCRIPTS:
    print(f'\n=== {script} ===')
    runpy.run_path(os.path.join(HERE, script), run_name='__main__')
