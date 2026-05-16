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
    FS = 28 
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off() 

    # 1. Draw the Manifold Surface (Extended surface to cover B_new)
    u = np.linspace(0, np.pi/2.1, 60)
    v = np.linspace(0, np.pi/1.8, 60)
    x = np.outer(np.sin(u), np.cos(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.cos(u), np.ones_like(v))
    
    ax.plot_surface(x, y, z, color='#B0B0B0', alpha=0.5, edgecolor='none', shade=True)

    # 2. Define our point B on the manifold
    B = np.array([np.sin(np.pi/4)*np.cos(np.pi/4), np.sin(np.pi/4)*np.sin(np.pi/4), np.cos(np.pi/4)])
    
    # 3. Draw the Tangent Space (Reverted bounds to 0.45)
    d = -B.dot(B)
    xx, yy = np.meshgrid(np.linspace(B[0]-0.45, B[0]+0.45, 2), np.linspace(B[1]-0.45, B[1]+0.45, 2))
    zz = (-B[0]*xx - B[1]*yy - d) / B[2]
    
    ax.plot_surface(xx, yy, zz, color='#F8F8F8', alpha=0.4, edgecolor='#666666', linewidth=1.5)

    # 4. Vectors
    G_offset = np.array([-0.25, 0.35, 0.85]) 
    G = B + G_offset
    G_vec = G - B
    
    # Calculate Riemannian Gradient
    proj_length = np.dot(G_vec, B)
    riemannian_vec_full = G_vec - proj_length * B
    
    # Exaggerated retraction
    scale_factor = 0.8  
    riemannian_vec = riemannian_vec_full * scale_factor
    G_riemannian = B + riemannian_vec

    # Plot Point B
    ax.scatter(*B, color='black', s=150, zorder=5)
    ax.text(B[0]+0.08, B[1]+0.0, B[2]+0.0, r'$\mathbf{B}$', fontsize=FS, fontweight='bold')

    # Plot Euclidean Gradient G
    # NOTE: linestyle='--' breaks 3D quiver arrowheads in matplotlib. 
    # Removed it and increased arrow_length_ratio so the head is clearly visible.
    ax.quiver(B[0], B[1], B[2], G_vec[0], G_vec[1], G_vec[2], color='gray', arrow_length_ratio=0.1, linewidth=4, linestyle='--')
    ax.text(G[0]+0.02, G[1]+0.02, G[2]+0.05, r'$\mathbf{G = \nabla_B W_2^2}$', fontsize=FS, color='black', fontweight='bold')

    # Plot Riemannian Gradient (Tangent Arrow)
    ax.quiver(B[0], B[1], B[2], riemannian_vec[0], riemannian_vec[1], riemannian_vec[2], color='black', arrow_length_ratio=0.1, linewidth=4)
    ax.text(G_riemannian[0]+0.08, G_riemannian[1]+0.02, G_riemannian[2]+0.05, r'$\mathbf{\nabla_{\mathcal{S}} B}$', fontsize=FS, color='black', fontweight='bold')

    # Retraction Step
    step_point = B + riemannian_vec 
    retracted_point = step_point / np.linalg.norm(step_point) 
    
    # Dotted drop line
    ax.plot([step_point[0], retracted_point[0]], [step_point[1], retracted_point[1]], [step_point[2], retracted_point[2]], color='black', linestyle=':', linewidth=3)
    ax.scatter(*retracted_point, color='black', s=120, zorder=5)
    ax.text(retracted_point[0]+0.05, retracted_point[1]+0.05, retracted_point[2]-0.08, r'$\mathbf{B_{new}}$', fontsize=FS, fontweight='bold')

    # Curved line (Spherical Interpolation)
    t_vals = np.linspace(0, 1, 30)
    theta = np.arccos(np.clip(np.dot(B, retracted_point), -1.0, 1.0))
    sin_theta = np.sin(theta)
    curve_pts = np.array([(np.sin((1-t)*theta)/sin_theta)*B + (np.sin(t*theta)/sin_theta)*retracted_point for t in t_vals])
    ax.plot(curve_pts[:,0], curve_pts[:,1], curve_pts[:,2], color='black', linestyle='-', linewidth=3.5, zorder=4)

    # --- LABELS ---
    ax.text(0.8, 0.9, 0.2, r'$\mathbf{T_B\mathcal{S}}$', fontsize=FS, horizontalalignment='center', fontweight='bold')
    
    corner_x, corner_y, corner_z = xx[1,0], yy[1,0], zz[1,0]
    ax.text(corner_x - 0.3, corner_y + 0.2, corner_z - 0.1, r'$\mathbf{\mathcal{S}}$' + ' (Orthogonal Matrices Space)', 
            fontsize=FS, horizontalalignment='center', fontweight='bold')

    # Set viewing angle
    ax.view_init(elev=25, azim=120)
    plt.tight_layout()
    
    file_name = 'stiefel_manifold_projection.pdf'
    plt.savefig(file_name, format='pdf', bbox_inches='tight')
    print(f"Saved {file_name}")
    plt.show()

# =====================================================================
# DIAGRAM 2: Discrete CDF Smoothed by Gaussian Dithering
# =====================================================================
def plot_dithering_cdf():
    # Massive figure for high-res PDF
    fig, ax = plt.subplots(figsize=(14, 10))
    
    n_points = 150
    discrete_data = np.random.choice([0, 1, 2, 3], size=n_points, p=[0.1, 0.4, 0.3, 0.2])
    discrete_data = np.sort(discrete_data)
    y_discrete = np.arange(1, n_points + 1) / n_points

    sigma_dither = 0.05
    dithered_data = discrete_data + np.random.normal(0, sigma_dither, size=n_points)
    dithered_data = np.sort(dithered_data)
    y_dithered = np.arange(1, n_points + 1) / n_points

    # Thickened lines
    ax.step(discrete_data, y_discrete, where='post', color='#666666', linewidth=4, label='Discrete Empirical CDF (Staircase)')
    ax.plot(dithered_data, y_dithered, color='black', linewidth=3.5, label='Dithered CDF (Continuous)')

    x_kernel = np.linspace(0.8, 1.2, 100)
    y_kernel = 0.5 + 0.035 * norm.pdf(x_kernel, loc=1, scale=sigma_dither)
    ax.plot(x_kernel, y_kernel, color='black', linestyle=':', linewidth=2.5)
    ax.fill_between(x_kernel, 0.5, y_kernel, color='#E0E0E0', alpha=0.7)
    
    # Scaled up math text
    ax.text(1.15, 0.65, r'$\sim \mathcal{N}(0, \sigma_{\text{dither}}^2)$', fontsize=24)
    ax.annotate('', xy=(2.0, 0.65), xytext=(2.0, 0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5, ls=':'))

    # Scaled up axes text
    ax.set_xlabel(r'Projected Value $y = \mathbf{w}^\top \mathbf{x}$', fontsize=26, labelpad=15)
    ax.set_ylabel(r'Cumulative Probability $F_{\mathbf{w}}(y)$', fontsize=26, labelpad=15)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 1.05)
    
    ax.set_xticks([0, 1, 2, 3])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(loc='lower right', fontsize=22)

    plt.tight_layout()
    
    file_name = 'gaussian_dithering_cdf.pdf'
    plt.savefig(file_name, format='pdf', bbox_inches='tight')
    print(f"Saved {file_name}")
    plt.close()

# Run the functions
plot_manifold_projection()
plot_dithering_cdf()