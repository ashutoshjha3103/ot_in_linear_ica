import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def set_thesis_theme():
    plt.rcParams.update({
        'figure.figsize': (8, 4),
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'xtick.bottom': False,
        'xtick.labelbottom': False,
        'ytick.left': False,
        'ytick.labelleft': False,
        'font.family': 'serif',
        'font.size': 12
    })
set_thesis_theme()

fig, ax = plt.subplots()

# Define the two distributions
x_mu = np.linspace(-4, 4, 500)
x_nu = np.linspace(4, 12, 500)

y_mu = norm.pdf(x_mu, 0, 1)
y_nu = norm.pdf(x_nu, 8, 1.2)

# Plot the distributions
ax.plot(x_mu, y_mu, color='#1f77b4', lw=2)
ax.fill_between(x_mu, 0, y_mu, color='#1f77b4', alpha=0.3)
ax.text(0, -0.05, r'Source Measure $\mu$', ha='center', fontsize=12)

ax.plot(x_nu, y_nu, color='#d62728', lw=2)
ax.fill_between(x_nu, 0, y_nu, color='#d62728', alpha=0.3)
ax.text(8, -0.05, r'Target Measure $\nu$', ha='center', fontsize=12)

# Draw the Transport Map arrow
arrow_start = 1.5
arrow_end = 6.5
ax.annotate('', xy=(arrow_end, 0.2), xytext=(arrow_start, 0.2),
            arrowprops=dict(arrowstyle="->", color="black", lw=2))

# Add the text and equation
ax.text(4, 0.22, 'Transport Map', ha='center', va='bottom', fontsize=12)
ax.text(4, 0.28, r'$\nu = T_{\#}\mu$', ha='center', va='bottom', fontsize=16)

ax.set_ylim(-0.1, 0.5)
ax.set_xlim(-5, 13)

plt.tight_layout()
plt.savefig('ot_pushforward_concept.png')
plt.show()