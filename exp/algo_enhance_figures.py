import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

# --- 1. Define the Cardoso / Mathematical Theme ---
def set_cardoso_theme():
    mpl.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',  # Computer Modern font (standard LaTeX)
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.grid': False,        # No grids for abstract math diagrams
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'figure.dpi': 300,
        'text.color': 'black',
        'axes.edgecolor': 'black'
    })

set_cardoso_theme()

# =====================================================================
# DIAGRAM 1: Riemannian Gradient on the Stiefel Manifold
# =====================================================================
def plot_manifold_projection():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off() # Hide standard 3D axes for a clean geometric sketch

    # 1. Draw the Manifold Surface (A curved sphere segment)
    u = np.linspace(0, np.pi/2.5, 50)
    v = np.linspace(0, np.pi/2.5, 50)
    x = np.outer(np.sin(u), np.cos(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.cos(u), np.ones_like(v))
    
    # Manifold surface: very faint and transparent
    ax.plot_surface(x, y, z, color='#B0B0B0', alpha=0.6, edgecolor='none', shade=True)

    # 2. Define our point W on the manifold
    W = np.array([np.sin(np.pi/4)*np.cos(np.pi/4), np.sin(np.pi/4)*np.sin(np.pi/4), np.cos(np.pi/4)])
    
    # 3. Draw the Tangent Space (A flat plane touching W)
    d = -W.dot(W)
    xx, yy = np.meshgrid(np.linspace(W[0]-0.4, W[0]+0.4, 2), np.linspace(W[1]-0.4, W[1]+0.4, 2))
    zz = (-W[0]*xx - W[1]*yy - d) / W[2]
    
    # CHANGED: Tangent plane is now stark white, high opacity, with a distinct edge
    ax.plot_surface(xx, yy, zz, color='#F8F8F8', alpha=0.3, edgecolor='#B0B0B0', linewidth=0.8)

    # 4. Vectors
    G = W + np.array([-0.1, 0.3, 0.4]) 
    
    G_vec = G - W
    proj_length = np.dot(G_vec, W)
    riemannian_vec = G_vec - proj_length * W
    G_riemannian = W + riemannian_vec

    # Plot Point W
    ax.scatter(*W, color='black', s=50, zorder=5)
    ax.text(W[0]+0.2, W[1]-0.1, W[2]-0.05, r'$\mathbf{W}$', fontsize=16)

    # Plot Euclidean Gradient G
    ax.quiver(W[0], W[1], W[2], G_vec[0], G_vec[1], G_vec[2], color='gray', arrow_length_ratio=0.1, linewidth=1.5, linestyle='--')
    ax.text(G[0]+0.02, G[1]+0.02, G[2]+0.02, r'$\mathbf{G} = \nabla_{\mathbf{W}} W_2^2$', fontsize=14, color='gray')

    # Plot Riemannian Gradient
    ax.quiver(W[0], W[1], W[2], riemannian_vec[0], riemannian_vec[1], riemannian_vec[2], color='black', arrow_length_ratio=0.1, linewidth=2)
    ax.text(G_riemannian[0] +0.05, G_riemannian[1]-0.12, G_riemannian[2]-0.05, r'$\nabla_{\mathcal{S}} \mathbf{W}$', fontsize=16, color='black')

    # Retraction Step 
    step_point = W + riemannian_vec * 0.8
    retracted_point = step_point / np.linalg.norm(step_point) 
    
    ax.plot([step_point[0], retracted_point[0]], [step_point[1], retracted_point[1]], [step_point[2], retracted_point[2]], color='black', linestyle=':', linewidth=2)
    ax.scatter(*retracted_point, color='black', s=30, zorder=5)
    ax.text(retracted_point[0]+0.05, retracted_point[1]+0.1, retracted_point[2]-0.05, r'$\mathbf{W}_{\text{new}}$', fontsize=14)

    # Label the spaces
    ax.text(0.1, 0.1, 0.9, r'$\mathcal{S}$ (Stiefel)', fontsize=14)
    ax.text(W[0]+0.2, W[1]-0.3, zz[1,1]+0.05, r'$T_{\mathbf{W}}\mathcal{S}$', fontsize=14)

    # CHANGED: Adjusted viewing angle for better depth perception
    ax.view_init(elev=20, azim=100)
    plt.tight_layout()
    plt.savefig('stiefel_manifold_projection.png', dpi=300, bbox_inches='tight')
    plt.show()

# =====================================================================
# DIAGRAM 2: Discrete CDF Smoothed by Gaussian Dithering
# =====================================================================
def plot_dithering_cdf():
    fig, ax = plt.figure(figsize=(8, 5)), plt.gca()
    
    n_points = 150
    discrete_data = np.random.choice([0, 1, 2, 3], size=n_points, p=[0.1, 0.4, 0.3, 0.2])
    discrete_data = np.sort(discrete_data)
    y_discrete = np.arange(1, n_points + 1) / n_points

    # REDUCED SIGMA FOR TIGHTER FIT
    sigma_dither = 0.05
    dithered_data = discrete_data + np.random.normal(0, sigma_dither, size=n_points)
    dithered_data = np.sort(dithered_data)
    y_dithered = np.arange(1, n_points + 1) / n_points

    ax.step(discrete_data, y_discrete, where='post', color='gray', linewidth=2.5, label='Discrete Empirical CDF (Staircase)')
    ax.plot(dithered_data, y_dithered, color='black', linewidth=2, label='Dithered CDF (Continuous)')

    # NARROWER VISUAL KERNEL
    x_kernel = np.linspace(0.8, 1.2, 100)
    # Scaled down to fit nicely above the step
    y_kernel = 0.5 + 0.035 * norm.pdf(x_kernel, loc=1, scale=sigma_dither)
    ax.plot(x_kernel, y_kernel, color='black', linestyle=':', linewidth=1.5)
    ax.fill_between(x_kernel, 0.5, y_kernel, color='#E0E0E0', alpha=0.5)
    ax.text(1.1, 0.65, r'$\sim \mathcal{N}(0, \sigma_{\text{dither}}^2)$', fontsize=12)
    
    ax.annotate('', xy=(2.0, 0.65), xytext=(2.0, 0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5, ls=':'))

    ax.set_xlabel('Projected Value $y = \mathbf{w}^\top \mathbf{x}$', fontsize=14)
    ax.set_ylabel('Cumulative Probability $F_{\mathbf{w}}(y)$', fontsize=14)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 1, 2, 3])
    ax.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.savefig('gaussian_dithering_cdf.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run the functions
plot_manifold_projection()
plot_dithering_cdf()
print("Saved 'stiefel_manifold_projection.png' and 'gaussian_dithering_cdf.png'")