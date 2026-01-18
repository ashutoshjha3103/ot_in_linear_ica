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
        self.W_white = torch.matmul(D_inv_sqrt, E.T)  # Store as class attribute
        self.X_white = torch.matmul(self.W_white, X_centered)
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

    def optimize_wasserstein2(self, prev_components=None, grid_points=100, continuous=True, max_iter=200, lr=0.1, n_restarts=3, decay_rate=0.5, decay_step=50):
        """
        Find ONE maximizer of Wasserstein-2 distance using Autograd.
        
        Includes stability fixes for high dimensions:
        1. Random Restarts: Avoids local optima.
        2. LR Scheduler: Decays LR to pinpoint exact peak.
        3. Hard Re-orthogonalization: Prevents numerical drift.
        """
        if continuous:
            # ==========================================================
            # CONTINUOUS MODE: Robust Gradient Ascent
            # (Includes Restarts, Annealing, Hard Re-orthogonalization)
            # ==========================================================
            best_w = None
            best_dist = -float('inf')

            for attempt in range(n_restarts):
                
                # 1. Initialize w randomly
                w = torch.randn(self.X.shape[0], device=self.X.device)
                
                # Initial Orthogonalization
                if prev_components is not None and prev_components.shape[0] > 0:
                    for pc in prev_components:
                        w = w - torch.dot(w, pc) * pc
                
                w = w / torch.norm(w)
                w.requires_grad_(True)
                
                current_lr = lr
                
                # Optimization Loop
                for i in range(max_iter):
                    
                    # LR Decay
                    if (i + 1) % decay_step == 0:
                        current_lr *= decay_rate
                    
                    # Forward
                    dist = self.wasserstein2_distance(w)
                    if w.grad is not None: w.grad.zero_()
                    dist.backward()
                    
                    grad = w.grad.data
                    
                    # Orthogonalize Gradient
                    if prev_components is not None and prev_components.shape[0] > 0:
                        for pc in prev_components:
                            grad = grad - torch.dot(grad, pc) * pc
                    
                    # Tangent Projection
                    grad = grad - torch.dot(grad, w.data) * w.data

                    # Gradient Clipping
                    grad_norm = torch.norm(grad)
                    if grad_norm > 1.0:
                        grad = grad / grad_norm

                    # Update
                    with torch.no_grad():
                        w.data = w.data + current_lr * grad
                        
                        # Hard Re-Orthogonalization
                        if prev_components is not None and prev_components.shape[0] > 0:
                            for pc in prev_components:
                                w.data = w.data - torch.dot(w.data, pc) * pc
                        
                        # Renormalize
                        w.data = w.data / torch.norm(w.data)
                
                # Keep Best Restart
                final_dist = self.wasserstein2_distance(w).item()
                if final_dist > best_dist:
                    best_dist = final_dist
                    best_w = w.detach().clone()
            
            return best_w, best_dist
        
        else:
            # ==========================================================
            # DISCRETE MODE: Grid Search (Restored)
            # ==========================================================
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
    
    # ========================================================
    # NEW METHODS: Symmetric Optimization
    # ========================================================

    def _symmetric_decorrelation(self, W):
        """
        Orthogonalize the rows of W simultaneously using (WW^T)^(-1/2).
        This forces the rows to be orthogonal to each other while minimally changing their directions.
        Formula: W_new = (WW^T)^(-1/2) * W
        """
        # 1. Compute Correlation Matrix: M = W * W^T
        M = torch.mm(W, W.t())
        
        # 2. Eigen Decomposition of M (Symmetric Positive Definite)
        evals, evecs = torch.linalg.eigh(M)
        
        # 3. Inverse Square Root: M^(-1/2) = E * D^(-1/2) * E^T
        # Add epsilon for numerical stability
        d_inv_sqrt = torch.diag(1.0 / torch.sqrt(evals + 1e-5))
        inv_sqrt_M = torch.mm(torch.mm(evecs, d_inv_sqrt), evecs.t())
        
        # 4. Apply to W
        return torch.mm(inv_sqrt_M, W)

    def optimize_symmetric(self, n_components=None, max_iter=300, lr=0.1, init_w=None):
        """
        Finds ALL components simultaneously using Symmetric Orthogonalization.
        Can accept an initialization matrix (init_w) to refine existing solutions.
        
        Parameters:
        -----------
        n_components: int (optional)
            Number of components to extract. Defaults to input dimension.
        max_iter: int
            Maximum number of iterations.
        lr: float
            Learning rate.

        Returns:
        --------
        W_sphere: torch tensor (n_components x n_features)
            The learned unmixing matrix on the sphere (orthogonal rows).
        """
        if n_components is None:
            n_components = self.X.shape[0]

        # 1. Initialize
        if init_w is not None:
            # Clone to ensure we don't modify the input tensor directly
            W = init_w.clone().to(self.X.device)
            # Ensure it fits the shape
            if W.shape[0] != n_components:
                # Handle case where init_w has different count than requested
                # (Optional safety check, usually not needed if logic is correct)
                pass 
        else:
            W = torch.randn(n_components, self.X.shape[0], device=self.X.device)
            
        # Force initial orthogonality
        W = self._symmetric_decorrelation(W)
        W.requires_grad_(True)
        
        # --- Gradient Ascent Loop ---
        for i in range(max_iter):
            if W.grad is not None: W.grad.zero_()
            
            # 2. Compute Total Loss
            total_dist = 0
            for k in range(n_components):
                total_dist += self.wasserstein2_distance(W[k])
            
            loss = -total_dist
            loss.backward()
            
            with torch.no_grad():
                grad = W.grad
                
                # Gradient Clipping
                grad_norm = torch.norm(grad)
                if grad_norm > 1.0:
                    grad = grad / grad_norm
                
                W += lr * grad
                
                # 3. SYMMETRIC ORTHOGONALIZATION
                # This is the "Magic" step that corrects global alignment
                W = self._symmetric_decorrelation(W)
                
                W.requires_grad_(True)
                
        return W.detach()