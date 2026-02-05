import numpy as np
import torch
import scipy.stats

class WassersteinICA:
    def __init__(self, X):
        """
        Initialize the Wasserstein ICA model.
        X: torch tensor of mixed signals (shape: num_signals x num_samples)
        """
        self.X = X
        self.n = X.shape[1]
        self.whitened = False
        self.epsilon = 1e-7
    
    def whiten(self):
        """
        Whiten the data (zero mean, unit variance, uncorrelated).
        Stores whitening matrix self.W_white for later reconstruction.
        """
        X_centered = self.X - torch.mean(self.X, dim=1, keepdim=True)
        cov = torch.matmul(X_centered, X_centered.t()) / (self.n - 1)
        D, E = torch.linalg.eigh(cov)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-5))
        self.W_white = torch.matmul(D_inv_sqrt, E.T)
        self.X_white = torch.matmul(self.W_white, X_centered)
        self.whitened = True
    
    def _normal_quantile(self, q):
        """
        Compute inverse CDF of standard normal at quantile q.
        """
        q_np = q.cpu().numpy()
        inv_cdf = scipy.stats.norm.ppf(q_np)
        return torch.tensor(inv_cdf, dtype=torch.float32, device=q.device)
    
    def wasserstein2_distance(self, w):
        """
        Compute W2 distance between projection w^T X and N(0,1).
        Uses Hazen plotting position for quantile alignment.
        """
        assert self.whitened, "Call whiten() before computing distance."
        y = torch.mv(self.X_white.t(), w)
        sorted_y, _ = torch.sort(y)
        steps = torch.arange(1, self.n + 1, dtype=torch.float32, device=self.X.device)
        q = (steps - 0.5) / self.n
        F_n_inv = self._normal_quantile(q)
        return torch.mean((sorted_y - F_n_inv) ** 2)

    def wasserstein1_distance(self, w):
        """
        Compute W1 distance (mean absolute difference) vs N(0,1).
        """
        assert self.whitened, "Call whiten() before computing distance."
        y = torch.mv(self.X_white.t(), w)
        sorted_y, _ = torch.sort(y)
        steps = torch.arange(1, self.n + 1, dtype=torch.float32, device=self.X.device)
        q = (steps - 0.5) / self.n
        F_n_inv = self._normal_quantile(q)
        return torch.mean(torch.abs(sorted_y - F_n_inv))

    def _wasserstein2_gradient_approx(self, w, delta=1e-5):
        """
        Finite difference approximation of W2 gradient (for debugging).
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
        Find ONE maximizer of W2 distance (Deflationary).
        Supports robust continuous optimization (Restarts+Annealing)
        or legacy discrete grid search.
        """
        if continuous:
            best_w = None
            best_dist = -float('inf')
            
            # Restart loop to avoid local optima
            for attempt in range(n_restarts):
                w = torch.randn(self.X.shape[0], device=self.X.device)
                if prev_components is not None and prev_components.shape[0] > 0:
                    for pc in prev_components:
                        w = w - torch.dot(w, pc) * pc
                w = w / torch.norm(w)
                w.requires_grad_(True)
                current_lr = lr
                
                # Gradient Ascent
                for i in range(max_iter):
                    if (i + 1) % decay_step == 0: current_lr *= decay_rate
                    dist = self.wasserstein2_distance(w)
                    if w.grad is not None: w.grad.zero_()
                    dist.backward()
                    
                    grad = w.grad.data
                    if prev_components is not None and prev_components.shape[0] > 0:
                        for pc in prev_components:
                            grad = grad - torch.dot(grad, pc) * pc
                    grad = grad - torch.dot(grad, w.data) * w.data # Tangent proj
                    
                    grad_norm = torch.norm(grad)
                    if grad_norm > 1.0: grad = grad / grad_norm # Clipping

                    with torch.no_grad():
                        w.data = w.data + current_lr * grad
                        # Hard re-orthogonalization to prevent drift
                        if prev_components is not None and prev_components.shape[0] > 0:
                            for pc in prev_components:
                                w.data = w.data - torch.dot(w.data, pc) * pc
                        w.data = w.data / torch.norm(w.data)
                
                final_dist = self.wasserstein2_distance(w).item()
                if final_dist > best_dist:
                    best_dist = final_dist
                    best_w = w.detach().clone()
            return best_w, best_dist
        else:
            # Legacy Discrete Grid Search
            angles = torch.linspace(0, 2 * np.pi, steps=grid_points, device=self.X.device)
            candidates = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
            if prev_components is not None and prev_components.shape[0] > 0:
                proj = torch.matmul(candidates, prev_components.t())
                candidates = candidates - torch.matmul(proj, prev_components)
                norms = torch.norm(candidates, dim=1, keepdim=True)
                mask = norms.squeeze() > 1e-6
                candidates = candidates[mask]
                if candidates.shape[0] == 0: raise ValueError("No valid candidates.")
                candidates = candidates / norms[mask]
            
            dist_best = -np.inf
            w_best = None
            for w in candidates:
                d = self.wasserstein2_distance(w).item()
                if d > dist_best:
                    dist_best = d
                    w_best = w
            return w_best, dist_best

    def _symmetric_decorrelation(self, W):
        """
        Symmetric orthogonalization: W_new = (WW^T)^(-1/2) * W.
        Distributes error evenly across all components.
        """
        M = torch.mm(W, W.t())
        evals, evecs = torch.linalg.eigh(M)
        d_inv_sqrt = torch.diag(1.0 / torch.sqrt(evals + 1e-5))
        inv_sqrt_M = torch.mm(torch.mm(evecs, d_inv_sqrt), evecs.t())
        return torch.mm(inv_sqrt_M, W)

    def optimize_symmetric(self, n_components=None, max_iter=300, lr=1.0, init_w=None, 
                           optimizer='sgd', penalty_weight=10.0, use_sinkhorn=False, 
                           reg=0.01, sinkhorn_iter=50):
        """
        Modified to support Sinkhorn distance during the L-BFGS phase.
        use_sinkhorn: If True, uses the smooth Sinkhorn surface instead of sorting.
        reg: Regularization parameter for Sinkhorn.
        sinkhorn_iter: Number of Sinkhorn iterations (Lower = Faster/Approximate).
        """
        if n_components is None: n_components = self.X.shape[0]

        if init_w is not None:
            W = init_w.clone().to(self.X.device)
        else:
            W = torch.randn(n_components, self.X.shape[0], device=self.X.device)
            W = self._symmetric_decorrelation(W)
        W.requires_grad_(True)
        
        if optimizer == 'sgd':
            for i in range(max_iter):
                if W.grad is not None: W.grad.zero_()
                
                # Check metric choice
                if use_sinkhorn:
                    # Pass sinkhorn_iter here
                    total_dist = sum(self.sinkhorn_distance(W[k], reg=reg, n_iter=sinkhorn_iter) for k in range(n_components))
                else:
                    total_dist = sum(self.wasserstein2_distance(W[k]) for k in range(n_components))
                
                loss = -total_dist
                loss.backward()
                
                with torch.no_grad():
                    grad = W.grad
                    if torch.norm(grad) > 1.0: grad = grad / torch.norm(grad)
                    
                    W += lr * grad
                    # Hard Projection
                    W.data = self._symmetric_decorrelation(W)
                    W.requires_grad_(True) 

        elif optimizer == 'lbfgs':
            # Annealing Schedule
            penalties = [penalty_weight, penalty_weight * 100, penalty_weight * 10000, penalty_weight * 1000000]
            steps = max_iter // len(penalties)
            if steps < 5: steps = 5

            for p in penalties:
                optim = torch.optim.LBFGS([W], lr=lr, max_iter=steps, history_size=50, 
                                          line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-7)
                def closure():
                    if W.grad is not None: W.grad.zero_()
                    
                    # CHOOSE DISTANCE METRIC
                    if use_sinkhorn:
                        # Pass sinkhorn_iter here
                        total_dist = sum(self.sinkhorn_distance(W[k], reg=reg, n_iter=sinkhorn_iter) for k in range(n_components))
                    else:
                        total_dist = sum(self.wasserstein2_distance(W[k]) for k in range(n_components))
                    
                    gram = torch.mm(W, W.t())
                    identity = torch.eye(n_components, device=self.X.device)
                    loss = -total_dist + (p * torch.norm(gram - identity) ** 2)
                    loss.backward()
                    return loss
                
                try: optim.step(closure)
                except RuntimeError: break
            
            with torch.no_grad(): W.data = self._symmetric_decorrelation(W) 
                
        return W.detach()
    
    def sinkhorn_distance(self, w, reg=0.01, n_iter=50):
        """
        Compute Entropy-Regularized W2 distance (Sinkhorn) in Log-Space.
        reg: Smoothing parameter (epsilon). 
        n_iter: Number of Sinkhorn iterations.
        """
        assert self.whitened, "Call whiten() before computing distance."
        y = torch.mv(self.X_white.t(), w)
        
        # 1. Target: Gaussian Quantiles
        steps = torch.arange(1, self.n + 1, dtype=torch.float32, device=self.X.device)
        q = (steps - 0.5) / self.n
        target = self._normal_quantile(q)

        # 2. Compute Cost Matrix: C_ij = (y_i - target_j)^2
        C = (y.unsqueeze(1) - target.unsqueeze(0)) ** 2

        # 3. Log-Space Sinkhorn Iterations (Stability Fix)
        # We find vectors f and g such that the transport plan P = exp((f + g - C)/reg)
        f = torch.zeros(self.n, device=self.X.device)
        g = torch.zeros(self.n, device=self.X.device)
        
        # Log of marginals (uniform = -log(n))
        log_mu = -torch.log(torch.tensor(self.n, dtype=torch.float32, device=self.X.device))
        
        for _ in range(n_iter):
            # Update f: f = reg * (log_mu - logsumexp((g - C)/reg))
            f = reg * (log_mu - torch.logsumexp((g.unsqueeze(0) - C) / reg, dim=1))
            # Update g: g = reg * (log_mu - logsumexp((f - C)/reg))
            g = reg * (log_mu - torch.logsumexp((f.unsqueeze(1) - C) / reg, dim=0))
            
        # 4. Return total cost: sum(P * C)
        # In log space, the transport plan is exp((f+g-C)/reg)
        log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) / reg
        return torch.sum(torch.exp(log_P) * C)
    
    def optimize_symmetric_sinkhorn(self, n_components=None, max_iter=300, lr=1.0, init_w=None, reg=0.05):
        return self.optimize_symmetric(
            n_components=n_components, 
            max_iter=max_iter, 
            lr=lr, 
            init_w=init_w, 
            optimizer='lbfgs',
            use_sinkhorn=True, # Critical flag
            reg=reg            # Pass the reg parameter
        )