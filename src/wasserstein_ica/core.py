import numpy as np
import torch
import scipy.stats

class WassersteinICA:
    def __init__(self, X):
        """
        X: torch tensor of mixed signals (shape: num_signals x num_samples)
        """
        self.X = X
        self.n = X.shape[1]
        self.whitened = False
        self.epsilon = 1e-7
    
    def whiten(self):
        """
        Whiten the mixture signal: zero mean, unit variance, and uncorrelated.
        """
        # Center data
        X_centered = self.X - torch.mean(self.X, dim=1, keepdim=True)
        # Compute covariance
        cov = torch.matmul(X_centered, X_centered.t()) / (self.n - 1)
        # Eigen decomposition
        D, E = torch.linalg.eigh(cov)
        # Whitening matrix with numerical stability
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-5))
        W = torch.matmul(D_inv_sqrt, E.T)
        self.X_white = torch.matmul(W, X_centered)
        self.whitened = True
    
    def _normal_quantile(self, q):
        """
        Inverse CDF of standard normal distribution at quantile levels.
        q: torch tensor in [0,1]
        """
        q_np = q.cpu().numpy()
        inv_cdf = scipy.stats.norm.ppf(q_np)
        return torch.tensor(inv_cdf, dtype=torch.float32, device=q.device)
    
    def wasserstein2_distance(self, w):
        """
        Compute the Wasserstein-2 distance from projections w.X and N(0,1)
        
        w: torch tensor of shape (num_signals,), unit norm
        returns: scalar, Wasserstein-2 distance
        """
        assert self.whitened, "Call whiten() before computing Wasserstein distance."
        # Project data
        y = torch.mv(self.X_white.t(), w)
        
        # Sort the projected values
        sorted_y, _ = torch.sort(y)
        
        # 3. Correct Quantile Definition (Hazen Plotting Position)
        # This aligns the i-th sample with its expected probability mass
        steps = torch.arange(1, self.n + 1, dtype=torch.float32, device=self.X.device)
        q = (steps - 0.5) / self.n

        # Theoretical quantiles from N(0,1)
        F_n_inv = self._normal_quantile(q)
        
        # Wasserstein-2 distance (root mean square)
        return torch.mean((sorted_y - F_n_inv) ** 2)

    def wasserstein1_distance(self, w):
        """
        Compute the Wasserstein-1 distance from projections w.X and N(0,1)
        
        w: torch tensor of shape (num_signals,), unit norm
        returns: scalar, Wasserstein-1 distance
        """
        assert self.whitened, "Call whiten() before computing Wasserstein distance."
        # Project data
        y = torch.mv(self.X_white.t(), w)
        
        # Sort the projected values
        sorted_y, _ = torch.sort(y)
        
        # 3. Correct Quantile Definition (Hazen Plotting Position)
        # This aligns the i-th sample with its expected probability mass
        steps = torch.arange(1, self.n + 1, dtype=torch.float32, device=self.X.device)
        q = (steps - 0.5) / self.n
        
        # Theoretical quantiles from N(0,1)
        F_n_inv = self._normal_quantile(q)
        
        # Wasserstein-1 distance (mean absolute difference)
        return torch.mean(torch.abs(sorted_y - F_n_inv))

    def _wasserstein2_gradient_approx(self, w, delta=1e-5):
        """
        Approximate gradient of Wasserstein-2 distance using finite differences.
        
        w: torch tensor of shape (num_signals,), unit norm
        delta: perturbation size for finite difference
        returns: gradient vector
        """
        grad = torch.zeros_like(w)
        base_val = self.wasserstein2_distance(w)
        for i in range(len(w)):
            w_perturb = w.clone()
            w_perturb[i] += delta
            w_perturb /= torch.norm(w_perturb)
            val = self.wasserstein2_distance(w_perturb)
            grad[i] = (val - base_val) / delta
        return grad

    def optimize_wasserstein2(self, prev_components=None, grid_points=100, continuous=True, max_iter=200, lr=0.1):
        """
        Find one maximizer of Wasserstein-2 distance over unit sphere using Autograd.
        
        Parameters:
        -----------
        prev_components: None or torch tensor
            Previously extracted components to enforce orthogonality
        grid_points: int
            Number of discretization points (for discrete mode)
        continuous: bool
            If True, uses Autograd SGD; if False, uses grid search
        max_iter: int
            Maximum number of iterations for SGD
        lr: float
            Learning rate for SGD

        Returns:
        --------
        w_best: torch tensor
            Optimal direction maximizing Wasserstein-2 distance
        dist_best: float
            Corresponding maximal Wasserstein-2 distance
        """
        if continuous:
            # --- Initialization ---
            # Initialize w randomly
            w = torch.randn(self.X.shape[0], device=self.X.device)
            
            # Orthogonalize w w.r.t. prev_components
            if prev_components is not None and prev_components.shape[0] > 0:
                for pc in prev_components:
                    w = w - torch.dot(w, pc) * pc
            
            # Normalize and enable gradient tracking
            w = w / torch.norm(w)
            w.requires_grad_(True)

            # --- Gradient Ascent ---
            for i in range(max_iter):
                # 1. Forward Pass: Compute distance
                # Note: We minimize negative distance to maximize distance
                dist = self.wasserstein2_distance(w)
                
                # 2. Backward Pass: Compute exact gradients
                if w.grad is not None:
                    w.grad.zero_()
                dist.backward()
                
                # 3. Get Gradient (detached from graph)
                grad = w.grad.data
                
                # 4. Enforce Orthogonality constraints on the Gradient
                if prev_components is not None and prev_components.shape[0] > 0:
                    for pc in prev_components:
                        grad = grad - torch.dot(grad, pc) * pc
                
                # 5. Project gradient onto the tangent space of the sphere
                # Remove the component of the gradient that points parallel to w
                # This ensures only the tangential component remains
                # Which makes sure the updates stay on the unit sphere.
                grad = grad - torch.dot(grad, w.data) * w.data

                # Normalize gradient for stable updates
                #############################FIX ###########################33
                #############################################################
                grad_norm = torch.norm(grad)
                if grad_norm > 1e-10: # larger threshold otherwise normalization will happen even if I am at minima
                    grad = grad / grad_norm # issue Michel: probably not necessary, explore (already on unit aphere)

                # 6. Update Step (Projected Gradient Ascent)
                with torch.no_grad():
                    w.data = w.data + lr * grad
                    w.data = w.data / torch.norm(w.data)
            
            # Final Return
            return w.detach(), self.wasserstein2_distance(w).item()
        
        else:
            # --- Discrete Grid Search (Unchanged) ---
            # Grid search method for discrete case (only works in 2D)
            angles = torch.linspace(0, 2 * np.pi, steps=grid_points, device=self.X.device)
            candidates = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

            if prev_components is not None and prev_components.shape[0] > 0:
                proj = torch.matmul(candidates, prev_components.t())
                candidates = candidates - torch.matmul(proj, prev_components)
                norms = torch.norm(candidates, dim=1, keepdim=True)
                
                mask = norms.squeeze() > 1e-6
                candidates = candidates[mask]
                if candidates.shape[0] == 0:
                    raise ValueError("No valid candidates after orthogonalization.")
                candidates = candidates / norms[mask]
            
            dist_best = -np.inf
            w_best = None
            for w in candidates:
                dist = self.wasserstein2_distance(w)
                dist_val = dist.item()
                if dist_val > dist_best:
                    dist_best = dist_val
                    w_best = w
            
            return w_best, dist_best