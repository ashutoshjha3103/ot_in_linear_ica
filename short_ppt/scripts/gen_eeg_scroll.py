"""
eeg_scroll.mp4 -- scrolling-playhead animation built on the *exact* MNE +
WassersteinICA pipeline used in exp/tpm2026_figures.py (CELL 11): frontal
channels of the MNE sample dataset, OT-ICA extraction, kurtosis-based
artifact identification, cleaned reconstruction. The expensive OT-ICA
optimization runs ONCE; the animation only sweeps a playhead + live RMS
readout across the precomputed raw/components/cleaned traces, so re-running
OT-ICA per frame is avoided.

v2: the result callouts are now PERSISTENT annotations pointing at the
blink window in panels (a) and (c), not text that only appears while the
playhead happens to be crossing that ~0.5s window during a 10s sweep --
that first version was on screen too briefly to actually read. The
playhead also slows way down specifically while crossing the blink
window (three-phase pacing: normal / slow / normal) so there's real time
to look at it, instead of a single uniform-speed sweep.
"""
import os
import sys
import numpy as np
import scipy.stats
import torch
import mne
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(__file__))
from common import set_deck_theme, save_animation, PALETTE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
from wasserstein_ica import WassersteinICA

OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'media', 'eeg_scroll.mp4')
set_deck_theme()

device = torch.device('cpu')

print('Loading MNE sample dataset...')
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)
raw.crop(tmin=10.0, tmax=20.0)

frontal_channels = ['EEG 001', 'EEG 002', 'EEG 003', 'EEG 004', 'EEG 005']
raw.pick_channels(frontal_channels)

X_eeg = raw.get_data()
time = raw.times
sfreq = raw.info['sfreq']
dim_eeg = X_eeg.shape[0]
X_eeg = (X_eeg - X_eeg.mean(1, keepdims=True)) / X_eeg.std(1, keepdims=True)
print(f'EEG shape: {X_eeg.shape} ({X_eeg.shape[1]} samples @ {sfreq:.0f} Hz)')

# ---- Run OT-ICA (identical hyperparameters to the paper figure) -----------
X_t_eeg = torch.tensor(X_eeg, dtype=torch.float32, device=device)
model_eeg = WassersteinICA(X_t_eeg)
model_eeg.whiten()

extracted_eeg = []
for _ in range(dim_eeg):
    prev = torch.stack(extracted_eeg) if extracted_eeg else None
    w, _ = model_eeg.optimize_wasserstein2(
        prev_components=prev, max_iter=200, n_restarts=50, dither_sigma=0.01)
    extracted_eeg.append(w)

W_eeg = model_eeg.optimize_symmetric(
    n_components=dim_eeg, max_iter=400, lr=0.05,
    init_w=torch.stack(extracted_eeg),
    optimizer='stiefel', batch_size=512, dither_sigma=0.01)

W_eeg_np = W_eeg.cpu().numpy()
W_white_np = model_eeg.W_white.cpu().numpy()
Z_hat = W_eeg_np @ model_eeg.X_white.cpu().numpy()

kurtoses = scipy.stats.kurtosis(Z_hat, axis=1)
artifact_idx = int(np.argmax(kurtoses))
kurtosis_artifact = float(kurtoses[artifact_idx])
kurtosis_others = float(np.mean([kurtoses[i] for i in range(dim_eeg) if i != artifact_idx]))
print(f'Artifact component (Comp {artifact_idx + 1}) kurtosis: {kurtosis_artifact:.2f}')
print(f'Mean kurtosis of remaining components: {kurtosis_others:.2f}')

peak_sample = int(np.argmax(np.abs(Z_hat[artifact_idx])))
if Z_hat[artifact_idx, peak_sample] < 0:
    Z_hat[artifact_idx] = -Z_hat[artifact_idx]

W_total_eeg = W_eeg_np @ W_white_np
W_total_inv = np.linalg.inv(W_total_eeg)
Z_cleaned = Z_hat.copy()
Z_cleaned[artifact_idx] = 0.0
X_cleaned = W_total_inv @ Z_cleaned

hw = int(0.25 * sfreq)
win = slice(max(0, peak_sample - hw), min(X_eeg.shape[1], peak_sample + hw))
rms_before = float(np.sqrt(np.mean(X_eeg[:, win] ** 2)))
rms_after = float(np.sqrt(np.mean(X_cleaned[:, win] ** 2)))
pct_reduction = (1.0 - rms_after / rms_before) * 100.0
print(f'RMS reduction in +-250ms blink window: {pct_reduction:.1f}%')

# ---- Animation: sweep a playhead across the precomputed traces ------------
OFFSET = 5.5
comp_colors = [PALETTE['OT-ICA'] if i == artifact_idx else '#444444' for i in range(dim_eeg)]
comp_labels = [f'Comp {i + 1}' for i in range(dim_eeg)]

fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.6), sharex=True)
fig.subplots_adjust(left=0.12, right=0.96, top=0.85, bottom=0.08, hspace=0.5)
fig.suptitle('EEG Artifact Removal: Isolating Ocular Blinks via Non-Gaussianity',
             fontsize=13.5, fontweight='bold', y=0.975)

for i in range(dim_eeg):
    axes[0].plot(time, X_eeg[i] - i * OFFSET, color='#444444', linewidth=0.9)
axes[0].set_yticks([-OFFSET * i for i in range(dim_eeg)])
axes[0].set_yticklabels(frontal_channels)
axes[0].set_title('(a) Raw frontal EEG', fontsize=12)

for i in range(dim_eeg):
    axes[1].plot(time, Z_hat[i] - i * OFFSET, color=comp_colors[i], linewidth=0.9)
axes[1].set_yticks([-OFFSET * i for i in range(dim_eeg)])
axes[1].set_yticklabels(comp_labels)
axes[1].set_title(f'(b) OT-ICA components  [Comp {artifact_idx + 1}: $\\kappa$={kurtosis_artifact:.1f}; '
                   f'others mean $\\kappa$={kurtosis_others:.1f}]', fontsize=12)

for i in range(dim_eeg):
    axes[2].plot(time, X_cleaned[i] - i * OFFSET, color='#444444', linewidth=0.9)
axes[2].set_yticks([-OFFSET * i for i in range(dim_eeg)])
axes[2].set_yticklabels(frontal_channels)
axes[2].set_title(f'(c) Cleaned EEG  [{pct_reduction:.0f}% RMS reduction in '
                   f'$\\pm$250ms blink window]', fontsize=12)
axes[2].set_xlabel('Time (s)')

blink_t0, blink_t1 = time[win.start], time[min(win.stop, len(time) - 1)]
blink_mid = (blink_t0 + blink_t1) / 2
for ax in axes:
    ax.axvspan(blink_t0, blink_t1, color=PALETTE['OT-ICA'], alpha=0.10, zorder=0)

# No on-screen callout text here by design -- the kurtosis and RMS-reduction
# numbers are already covered by the narration and by the panel titles
# above (b) and (c); a redundant annotation was found unnecessary.

playheads = [ax.axvline(time[0], color=PALETTE['FastICA'], lw=2, zorder=5) for ax in axes]
clock = fig.text(0.5, 0.895, '', ha='center', fontsize=11, color='#555555')

# Three-phase pacing: normal speed up to the blink window, a deliberate slow
# crawl THROUGH the window (most of the screen time despite being the
# shortest real-time span), then normal speed to the end -- instead of one
# uniform-speed sweep that blows past the blink in a fraction of a second.
idx_before = np.linspace(0, win.start, 90).astype(int)
idx_window = np.linspace(win.start, win.stop, 110).astype(int)
idx_after = np.linspace(win.stop, len(time) - 1, 60).astype(int)
sweep_idx = np.concatenate([idx_before, idx_window, idx_after])
N_FRAMES = len(sweep_idx)


def update(frame):
    idx = sweep_idx[frame]
    t = time[idx]
    for ph in playheads:
        ph.set_xdata([t, t])
    clock.set_text(f't = {t:5.2f}s')
    return playheads + [clock]


anim = FuncAnimation(fig, update, frames=N_FRAMES, blit=False)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_animation(anim, OUT, fps=20)
print('wrote', OUT)
