import numpy as np 
import torch 
import scipy.stats 

class WassersteinICA: 
    def __init__(self, X): 
        self.X = X 
        self.n = X.shape[1] 
        self.whitened = False 
        self.epsilon = 1e-7 
        # Cache for the analytical target (computed once) 
        self.analytical_target = None  

    def whiten(self): 
        """ 
        Whiten the data (zero mean, unit variance, uncorrelated). 
        """ 
        X_centered = self.X - torch.mean(self.X, dim=1, keepdim=True) 
        cov = torch.matmul(X_centered, X_centered.t()) / (self.n - 1) 
        D, E = torch.linalg.eigh(cov) 
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-5)) 
        self.W_white = torch.matmul(D_inv_sqrt, E.T) 
        self.X_white = torch.matmul(self.W_white, X_centered) 
        self.whitened = True 
        # Pre-compute the exact analytical Gaussian target 
        self.analytical_target = self._compute_analytical_target(self.n) 
      
    def _compute_analytical_target(self, n): 
        """ 
        Computes the EXACT 'Average Quantile' for each bin analytically. 
        Formula: Target_i = N * (pdf(z_{i-1}) - pdf(z_i)) 
        This replaces 'sampling' with exact calculus. 
        """ 
        p_edges = np.linspace(0, 1, n + 1) 
        z_edges = scipy.stats.norm.ppf(p_edges) 
        phi_edges = scipy.stats.norm.pdf(z_edges) 
          
        target_np = n * (phi_edges[:-1] - phi_edges[1:]) 
        return torch.tensor(target_np, dtype=torch.float32, device=self.X.device) 

    # ========================================== 
    # VECTORIZED: Core Distance Metric 
    # ========================================== 
    def wasserstein2_analytical(self, W, cost='l2', dither_sigma=0.0): 
        """ 
        Computes W distance.  
        Supports both single vectors (legacy) and matrices (batched parallel). 
        cost: 'l2' for standard Wasserstein, 'logcosh' for robust Huber-like geometry.
        dither_sigma: Injects continuous noise to smooth discrete CDF steps.
        """ 
        assert self.whitened, "Call whiten() before computing distance." 
          
        is_1d = W.dim() == 1 
        if is_1d: 
            W = W.unsqueeze(0) 
              
        Y = torch.mm(W, self.X_white)  
        
        # DITHERING: Inject continuous noise to break discrete ties and smooth the CDF
        if dither_sigma > 0:
            Y = Y + torch.randn_like(Y) * dither_sigma
            
        sorted_Y, _ = torch.sort(Y, dim=1)  
        diff = sorted_Y - self.analytical_target 
          
        if cost == 'l2':
            distances = torch.mean(diff ** 2, dim=1) 
        elif cost == 'logcosh':
            # Numerically stable logcosh to prevent NaN gradients on massive outliers
            abs_diff = torch.abs(diff)
            logcosh_diff = abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff)) - np.log(2.0)
            distances = torch.mean(logcosh_diff, dim=1)
        else:
            raise ValueError("cost must be 'l2' or 'logcosh'")
          
        if is_1d: 
            return distances[0] 
        return distances

    # ========================================== 
    # VECTORIZED: Phase 1 (Deflation & Restarts) 
    # ========================================== 
    def optimize_wasserstein2(self, prev_components=None, grid_points=100, continuous=True, 
                              max_iter=200, lr=0.1, n_restarts=50, decay_rate=0.5, decay_step=50, cost='l2', dither_sigma=0.0): 
        """ 
        Find ONE maximizer of W distance (Deflationary). 
        """ 
        if continuous: 
            W_batch = torch.randn(n_restarts, self.X.shape[0], device=self.X.device) 
              
            if prev_components is not None and prev_components.shape[0] > 0: 
                proj = torch.matmul(W_batch, prev_components.t()) 
                W_batch = W_batch - torch.matmul(proj, prev_components) 
                  
            W_batch = W_batch / torch.norm(W_batch, dim=1, keepdim=True) 
            W_batch.requires_grad_(True) 
            current_lr = lr 
              
            for i in range(max_iter): 
                if (i + 1) % decay_step == 0: current_lr *= decay_rate 
                  
                # Pass the dither parameter down
                dist = self.wasserstein2_analytical(W_batch, cost=cost, dither_sigma=dither_sigma).sum() 
                  
                if W_batch.grad is not None: W_batch.grad.zero_() 
                dist.backward() 
                  
                with torch.no_grad(): 
                    grad = W_batch.grad 
                    if prev_components is not None and prev_components.shape[0] > 0: 
                        proj_grad = torch.matmul(grad, prev_components.t()) 
                        grad = grad - torch.matmul(proj_grad, prev_components) 
                          
                    dot_pw = torch.sum(grad * W_batch.data, dim=1, keepdim=True) 
                    grad = grad - dot_pw * W_batch.data  
                      
                    grad_norms = torch.norm(grad, dim=1, keepdim=True) 
                    grad = torch.where(grad_norms > 1.0, grad / grad_norms, grad) 

                    W_batch.data += current_lr * grad 
                      
                    if prev_components is not None and prev_components.shape[0] > 0: 
                        proj = torch.matmul(W_batch.data, prev_components.t()) 
                        W_batch.data = W_batch.data - torch.matmul(proj, prev_components) 
                          
                    W_batch.data /= torch.norm(W_batch.data, dim=1, keepdim=True) 
              
            with torch.no_grad(): 
                # Evaluate final best vector without noise to get the true mathematical distance
                final_distances = self.wasserstein2_analytical(W_batch, cost=cost, dither_sigma=0.0) 
                best_idx = torch.argmax(final_distances) 
                best_w = W_batch[best_idx].detach().clone() 
                best_dist = final_distances[best_idx].item() 
                  
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
                d = self.wasserstein2_analytical(w, cost=cost, dither_sigma=dither_sigma).item() 
                if d > dist_best: 
                    dist_best = d 
                    w_best = w 
            return w_best, dist_best 

    def _symmetric_decorrelation(self, W): 
        M = torch.mm(W, W.t()) 
        evals, evecs = torch.linalg.eigh(M) 
        d_inv_sqrt = torch.diag(1.0 / torch.sqrt(evals + 1e-5)) 
        inv_sqrt_M = torch.mm(torch.mm(evecs, d_inv_sqrt), evecs.t()) 
        return torch.mm(inv_sqrt_M, W) 

    # ========================================== 
    # VECTORIZED: Phase 2 
    # ========================================== 
    def optimize_symmetric(self, n_components=None, max_iter=300, lr=1.0, init_w=None,  
                           optimizer='sgd', penalty_weight=10.0, use_sinkhorn=False,  
                           reg=0.01, sinkhorn_iter=50, cost='l2', dither_sigma=0.0, batch_size=512): 
        if n_components is None: n_components = self.X.shape[0] 

        if init_w is not None: 
            W = init_w.clone().to(self.X.device) 
        else: 
            W = torch.randn(n_components, self.X.shape[0], device=self.X.device) 
            W = self._symmetric_decorrelation(W) 
        W.requires_grad_(True) 
          
        # Store originals to safely patch during stochastic batching
        original_X_white = self.X_white
        original_n = self.n
        original_target = self.analytical_target
          
        if optimizer == 'sgd': 
            for i in range(max_iter): 
                if W.grad is not None: W.grad.zero_() 
                  
                if use_sinkhorn: 
                    total_dist = self.sinkhorn_distance(W, reg=reg, n_iter=sinkhorn_iter).sum() 
                else: 
                    total_dist = self.wasserstein2_analytical(W, cost=cost, dither_sigma=dither_sigma).sum() 
                  
                loss = -total_dist 
                loss.backward() 
                  
                with torch.no_grad(): 
                    grad = W.grad 
                    grad_norms = torch.norm(grad, dim=1, keepdim=True) 
                    grad = torch.where(grad_norms > 1.0, grad / grad_norms, grad) 
                      
                    W += lr * grad 
                    W.data = self._symmetric_decorrelation(W) 
                    W.requires_grad_(True)  

        elif optimizer == 'stiefel':
            current_lr = lr
            for i in range(max_iter):
                if W.grad is not None: W.grad.zero_()
                
                # STOCHASTIC BATCHING: Randomly slice data to inject gradient noise
                if batch_size is not None and batch_size < original_n:
                    indices = torch.randperm(original_n, device=self.X.device)[:batch_size]
                    self.X_white = original_X_white[:, indices]
                    self.n = batch_size
                    self.analytical_target = self._compute_analytical_target(self.n)
                
                if use_sinkhorn:
                    total_dist = self.sinkhorn_distance(W, reg=reg, n_iter=sinkhorn_iter).sum()
                else:
                    total_dist = self.wasserstein2_analytical(W, cost=cost, dither_sigma=dither_sigma).sum()
                
                total_dist.backward()
                
                with torch.no_grad():
                    grad = W.grad
                    
                    # Stiefel Projection
                    G_Wt = torch.mm(grad, W.data.t())
                    W_Gt = torch.mm(W.data, grad.t())
                    sym = 0.5 * (G_Wt + W_Gt)
                    tangent_grad = grad - torch.mm(sym, W.data)
                    
                    tangent_norms = torch.norm(tangent_grad, dim=1, keepdim=True)
                    tangent_grad = torch.where(tangent_norms > 1.0, tangent_grad / tangent_norms, tangent_grad)
                    
                    # Apply step with decaying learning rate to allow settling
                    W += current_lr * tangent_grad
                    W.data = self._symmetric_decorrelation(W.data)
                    W.requires_grad_(True)
                    
                # Decay learning rate by 1% each step
                current_lr *= 0.99 

        elif optimizer == 'lbfgs': 
            penalties = [penalty_weight, penalty_weight * 100, penalty_weight * 10000, penalty_weight * 1000000] 
            steps = max_iter // len(penalties) 
            if steps < 5: steps = 5 

            for p in penalties: 
                optim = torch.optim.LBFGS([W], lr=lr, max_iter=steps, history_size=50,  
                                          line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-7) 
                def closure(): 
                    if W.grad is not None: W.grad.zero_() 
                      
                    if use_sinkhorn: 
                        total_dist = self.sinkhorn_distance(W, reg=reg, n_iter=sinkhorn_iter).sum() 
                    else: 
                        total_dist = self.wasserstein2_analytical(W, cost=cost, dither_sigma=dither_sigma).sum() 
                      
                    gram = torch.mm(W, W.t()) 
                    trace_gram = torch.trace(gram) 
                    trace_gram_sq = torch.trace(torch.mm(gram, gram)) 
                    ortho_penalty = trace_gram_sq - 2 * trace_gram + n_components 
                      
                    loss = -total_dist + (p * ortho_penalty) 
                    loss.backward() 
                    return loss 
                  
                try: optim.step(closure) 
                except RuntimeError: break 
              
            with torch.no_grad(): W.data = self._symmetric_decorrelation(W)  
        
        # Restore original full dataset after optimization loop finishes
        self.X_white = original_X_white
        self.n = original_n
        self.analytical_target = original_target
                  
        return W.detach()
      

    # ========================================== 
    # NEW: OT-Mapping Fixed-Point Rule 
    # ========================================== 
    def optimize_fixed_point(self, n_components=None, max_iter=100, tol=1e-5, init_w=None, step_size=0.5): 
        """ 
        Calculates the OT mapping to the perfect Gaussian, then steps AWAY from it. 
        Acts as Gradient Ascent on the Wasserstein landscape. 
        """ 
        assert self.whitened, "Call whiten() before optimization." 
        if n_components is None: n_components = self.X.shape[0] 

        # 1. Initialization 
        if init_w is not None: 
            W = init_w.clone().to(self.X.device) 
        else: 
            W = torch.randn(n_components, self.X.shape[0], device=self.X.device) 
          
        W = self._symmetric_decorrelation(W) 
          
        # We need target matrix T broadcasted to match dimensions (C x N) 
        T = self.analytical_target.unsqueeze(0).expand(n_components, -1) 
          
        for i in range(max_iter): 
            # Step 1: Project the data (Y = WX) 
            Y = torch.mm(W, self.X_white) 
              
            # Step 2: Find the ranking/sorting indices 
            idx = torch.argsort(Y, dim=1) 
              
            # Step 3: Create the "Ideal Target" (Y_ideal) 
            Y_ideal = torch.empty_like(Y) 
            Y_ideal.scatter_(1, idx, T) 
              
            # Step 4: The Gradient (Direction pointing INTO the Gaussian valley) 
            G = torch.mm(Y_ideal, self.X_white.t()) / (self.n - 1) 
              
            # Step 5: The Anti-Gaussian Step (Climbing the hill) 
            # We subtract G to step AWAY from the Gaussian 
            W_new = W - step_size * G 
              
            # Step 6: Symmetrically Orthogonalize W_new 
            W_new = self._symmetric_decorrelation(W_new) 
              
            # Step 7: Check for convergence 
            cos_theta = torch.abs(torch.diag(torch.mm(W_new, W.t()))) 
            min_cos = torch.min(cos_theta).item() 
              
            W = W_new 
              
            if (1.0 - min_cos) < tol: 
                break 
                  
        return W.detach() 

    # ========================================== 
    # LEGACY / BACKWARD COMPATIBILITY FUNCTIONS 
    # ========================================== 
    def _normal_quantile(self, q): 
        q_np = q.cpu().numpy() 
        inv_cdf = scipy.stats.norm.ppf(q_np) 
        return torch.tensor(inv_cdf, dtype=torch.float32, device=q.device) 
      
    def wasserstein2_distance(self, w): 
        assert self.whitened, "Call whiten() before computing distance." 
        y = torch.mv(self.X_white.t(), w) 
        sorted_y, _ = torch.sort(y) 
        steps = torch.arange(1, self.n + 1, dtype=torch.float32, device=self.X.device) 
        q = (steps - 0.5) / self.n 
        F_n_inv = self._normal_quantile(q) 
        return torch.mean((sorted_y - F_n_inv) ** 2) 

    def wasserstein1_distance(self, w): 
        assert self.whitened, "Call whiten() before computing distance." 
        y = torch.mv(self.X_white.t(), w) 
        sorted_y, _ = torch.sort(y) 
        steps = torch.arange(1, self.n + 1, dtype=torch.float32, device=self.X.device) 
        q = (steps - 0.5) / self.n 
        F_n_inv = self._normal_quantile(q) 
        return torch.mean(torch.abs(sorted_y - F_n_inv)) 

    def _wasserstein2_gradient_approx(self, w, delta=1e-5): 
        grad = torch.zeros_like(w) 
        base_val = self.wasserstein2_distance(w) 
        for i in range(len(w)): 
            w_perturb = w.clone() 
            w_perturb[i] += delta 
            w_perturb /= torch.norm(w_perturb) 
            val = self.wasserstein2_distance(w_perturb) 
            grad[i] = (val - base_val) / delta 
        return grad 

    def sinkhorn_distance(self, W, reg=0.01, n_iter=50): 
        """ 
        Batched Entropy-Regularized W2 distance (Sinkhorn) in Log-Space. 
        W shape: (n_components, n_dimensions) OR (n_dimensions,) 
        """ 
        assert self.whitened, "Call whiten() before computing distance." 
          
        is_1d = W.dim() == 1 
        if is_1d: 
            W = W.unsqueeze(0) 
              
        B = W.shape[0] # Batch size / Number of components 
          
        # 1. Project all data at once (Shape: B x N) 
        Y = torch.mm(W, self.X_white) 
          
        # 2. Target: Gaussian Quantiles (Shape: N) 
        steps = torch.arange(1, self.n + 1, dtype=torch.float32, device=self.X.device) 
        q = (steps - 0.5) / self.n 
        target = self._normal_quantile(q) 
          
        # 3. Batched Cost Matrix C: (B, N_y, N_target) 
        # Broadcasting: Y is (B, N, 1), target is (1, 1, N) 
        C = (Y.unsqueeze(2) - target.view(1, 1, self.n)) ** 2 
          
        # 4. Sinkhorn Iterations 
        f = torch.zeros(B, self.n, device=self.X.device) 
        g = torch.zeros(B, self.n, device=self.X.device) 
        log_mu = -torch.log(torch.tensor(self.n, dtype=torch.float32, device=self.X.device)) 
          
        for _ in range(n_iter): 
            # Update f: Sum over target dimension (dim=2) 
            f = reg * (log_mu - torch.logsumexp((g.unsqueeze(1) - C) / reg, dim=2)) 
            # Update g: Sum over Y dimension (dim=1) 
            g = reg * (log_mu - torch.logsumexp((f.unsqueeze(2) - C) / reg, dim=1)) 
              
        # 5. Calculate total cost for each batch element 
        # log_P shape: (B, N, N) 
        log_P = (f.unsqueeze(2) + g.unsqueeze(1) - C) / reg 
        distances = torch.sum(torch.exp(log_P) * C, dim=(1, 2)) 
          
        if is_1d: 
            return distances[0] 
        return distances 
      
    def optimize_symmetric_sinkhorn(self, n_components=None, max_iter=300, lr=1.0, init_w=None, reg=0.05): 
        return self.optimize_symmetric( 
            n_components=n_components,  
            max_iter=max_iter,  
            lr=lr,  
            init_w=init_w,  
            optimizer='lbfgs', 
            use_sinkhorn=True,  
            reg=reg             
        )