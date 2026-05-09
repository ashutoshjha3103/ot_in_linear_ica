import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def set_thesis_theme():
    plt.rcParams.update({
        'figure.figsize': (24, 12),    # Huge figure size
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
        'font.size': 24              # Doubled base font size
    })
set_thesis_theme()

fig, ax = plt.subplots()

# Define the two distributions
x_mu = np.linspace(-4, 4, 500)
x_nu = np.linspace(4, 12, 500)

y_mu = norm.pdf(x_mu, 0, 1)
y_nu = norm.pdf(x_nu, 8, 1.2)

# Set a massive line width
LW = 4.5

# Define the requested colors
color_mu = '#d62728' # Red for source
color_nu = '#0b5394' # Slightly darker blue for target

# Plot the Source distribution
ax.plot(x_mu, y_mu, color=color_mu, lw=LW)
ax.fill_between(x_mu, 0, y_mu, color=color_mu, alpha=0.3)
ax.text(0, -0.05, r'Source Measure $\mu$', ha='center', fontsize=42)

# Plot the Target distribution
ax.plot(x_nu, y_nu, color=color_nu, lw=LW)
ax.fill_between(x_nu, 0, y_nu, color=color_nu, alpha=0.3)
ax.text(8, -0.05, r'Target Measure $\nu$', ha='center', fontsize=42)

# Draw the Transport Map arrow
arrow_start = 1.5
arrow_end = 6.5
ax.annotate('', xy=(arrow_end, 0.2), xytext=(arrow_start, 0.2),
            arrowprops=dict(arrowstyle="->", color="black", lw=LW))

# Add the text and equation (Sizes doubled, Y-height slightly separated to prevent overlap)
ax.text(4, 0.22, 'Transport Map', ha='center', va='bottom', fontsize=42)
ax.text(4, 0.32, r'$\nu = T_{\#}\mu$', ha='center', va='bottom', fontsize=56) 

ax.set_ylim(-0.1, 0.5)
ax.set_xlim(-5, 13)

plt.tight_layout()

# Save as high-quality PDF
file_name = 'ot_pushforward_concept.pdf'
plt.savefig(file_name, format='pdf', bbox_inches='tight')
print(f"Successfully saved huge figure to {file_name}")

plt.show()