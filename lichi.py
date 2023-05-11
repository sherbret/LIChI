#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : LIChI
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

class LIChI(nn.Module):
    def __init__(self):
        super(LIChI, self).__init__()
        
    def set_parameters(self, sigma=25.0, constraints='affine', method='n2n',
                        p1=11, p2=6, k1=16, k2=64, w=65, s=3, M=9):
        self.sigma = sigma # sigma parameter for Gaussian noise
        self.p1 = p1 # patch size for step 1
        self.p2 = p2 # patch size for step 2
        self.k1 = k1 # group size for step 1
        self.k2 = k2 # group size for step 2
        self.M = M # number of iterations
        self.window = w # size of the window centered around reference patches within which similar patches are searched (odd number)
        self.step = s # moving step size from one reference patch to another
        self.constraints = constraints # either 'linear' or 'affine'
        self.method = method # either 'n2n', 'sure', 'avg' or 'noisy'
        
    @staticmethod
    def block_matching(input_x, k, p, w, s):
        if input_x.device == torch.device("cuda:0"): 
            input_x = input_x.half()

        v = w // 2
        w_large = 2*v + p
        input_x_pad = F.pad(input_x, [v]*4, mode='constant', value=float('inf'))
        N, C, H, W = input_x.size() 
        Href, Wref = -((H - p + 1) // -s), -((W - p + 1) // -s) # ceiling division, represents the number of reference patches along each axis for unfold with stride=s
        ind_H_ref = torch.arange(0, H-p+1, step=s, device=input_x.device)      
        ind_W_ref = torch.arange(0, W-p+1, step=s, device=input_x.device)
        if (H - p + 1) % s != 1:
            ind_H_ref = torch.cat((ind_H_ref, torch.tensor([H - p], device=input_x.device)), dim=0)
        if (W - p + 1) % s != 1:
            ind_W_ref = torch.cat((ind_W_ref, torch.tensor([W - p], device=input_x.device)), dim=0)
        ind_H_ref, ind_W_ref = ind_H_ref.view(1, -1, 1, 1), ind_W_ref.view(1, 1, -1, 1)
        ind_H_ref, ind_W_ref = ind_H_ref.expand(N, -1, ind_W_ref.size(2), k), ind_W_ref.expand(N, ind_H_ref.size(1), -1, k)
        
        def block_matching_aux(input_x, input_x_pad, k, p, v, s):
            N, C, H, W = input_x.size()
            w_large = 2*v + p
            Href, Wref = -((H - p + 1) // -s), -((W - p + 1) // -s) # ceiling division, represents the number of reference patches along each axis for unfold with stride=s
            ref_patches = F.unfold(input_x, p, stride=s).transpose(1, 2).reshape(N*Href*Wref, 1, p, p)
            local_windows = F.unfold(input_x_pad, w_large, stride=s).transpose(1, 2).reshape(1, N*Href*Wref, w_large, w_large)
            scalar_product = F.conv2d(local_windows, ref_patches / p**2, groups=N*Href*Wref) # assumes that N = 1
            norm_patches = F.avg_pool2d(local_windows**2, p, stride=1)
            distances = torch.nan_to_num(norm_patches - 2 * scalar_product, nan=float('inf'))
            distances[:, :, v, v] = -float('inf') # the reference patch is always taken
            distances = distances.view(N, Href*Wref, -1)
            indices = torch.topk(distances, k, dim=2, largest=False, sorted=False).indices.view(N, Href, Wref, k)
            return indices

        indices = torch.empty_like(ind_H_ref)
        indices[:, :Href, :Wref, :] = block_matching_aux(input_x, input_x_pad, k, p, v, s)
        if (H - p + 1) % s != 1:
            indices[:, Href:, :Wref, :] = block_matching_aux(input_x[:, :, -p:, :], input_x_pad[:, :, -w_large:, :], k, p, v, s)
        if (W - p + 1) % s != 1:
            indices[:, :Href, Wref:, :] = block_matching_aux(input_x[:, :, :, -p:], input_x_pad[:, :, :, -w_large:], k, p, v, s)
            if (H - p + 1) % s != 1:
                indices[:, Href:, Wref:, :] = block_matching_aux(input_x[:, :, -p:, -p:], input_x_pad[:, :, -w_large:, -w_large:], k, p, v, s)
                
        # (ind_row, ind_col) is a 2d-representation of indices
        ind_row = torch.div(indices, 2*v+1, rounding_mode='floor') - v
        ind_col = torch.fmod(indices, 2*v+1) - v
        
        # from 2d to 1d representation of indices 
        indices = ((ind_H_ref + ind_row) * (W-p+1) + (ind_W_ref + ind_col)).view(N, -1)
        return indices
    
    def compute_theta(self, Q, D):
        N, B, k, _ = Q.size()
        if self.constraints == 'linear' or self.constraints == 'affine':
            Ik = torch.eye(k, dtype=Q.dtype, device=Q.device).expand(N, B, -1, -1)
            L = torch.linalg.cholesky(Q)
            Qinv = torch.cholesky_solve(Ik, L)
            if self.constraints == 'linear':
                theta = Ik - Qinv * D.unsqueeze(-1)
            else:
                Qinv1 = torch.sum(Qinv, dim=3, keepdim=True)
                Qinv2 = torch.sum(Qinv1, dim=2, keepdim=True)
                theta = Ik - (Qinv - Qinv1 @ Qinv1.transpose(2, 3) / Qinv2) * D.unsqueeze(-1)
        else:
            raise ValueError('constraints must be either linear, affine, conical or convex.')
        return theta.transpose(2,3)
    
    @staticmethod 
    def gather_groups(input_y, indices, k, p):
        unfold_y = F.unfold(input_y, p)
        N, n, _ = unfold_y.size()
        Y = torch.gather(unfold_y, dim=2, index=indices.view(N, 1, -1).expand(-1, n, -1)).transpose(1, 2).view(N, -1, k, n)
        return Y

    def denoise1(self, Y, sigma):
        N, B, k, n = Y.size()
        if self.method=='sure':
            D = n * sigma**2 * torch.ones(1, 1, k, dtype=Y.dtype, device=Y.device)
            Q = Y @ Y.transpose(2, 3)
            theta = self.compute_theta(Q, D)
        elif self.method=='n2n':
            alpha = 0.5
            D = n * sigma**2 * torch.ones(1, 1, k, dtype=Y.dtype, device=Y.device)
            E = n * alpha**2 * sigma**2 * torch.ones(1, 1, k, dtype=Y.dtype, device=Y.device)
            Q = Y @ Y.transpose(2, 3)
            theta = self.compute_theta(Q + torch.diag_embed(E), D + E)
        elif self.method=='avg':
            theta = torch.ones(N, B, k, k, dtype=Y.dtype, device=Y.device) / k
        elif self.method=='noisy':
            theta = torch.eye(k, dtype=Y.dtype, device=Y.device).expand(N, B, -1, -1)  
        else:
            raise Exception('Method must be either sure, n2n, avg or noisy.')
        X_hat = theta @ Y 
        weights = 1 / torch.sum(theta**2, dim=3, keepdim=True).clip(1/k, 1)
        return X_hat, weights

    def denoise2(self, Z, X, Y, sigma, tau):
        N, B, k, n = Z.size()
        t = 1 - torch.std(Y - Z, dim=(2,3), keepdim=True, unbiased=False) / sigma
        t = t.clip(min=tau+1e-6)
        D = n * (sigma * t[:, :, :, 0].expand(-1, -1, k))**2
        Q = X @ X.transpose(2, 3) + torch.diag_embed(D)
        xi = self.compute_theta(Q, D)
        X_hat = xi @ Z
        Z_hat = (1 - tau/t) * X_hat + tau/t * Z
        weights = 1 / torch.sum(xi**2, dim=3, keepdim=True).clip(min=1/k)
        return X_hat, Z_hat, weights

    @staticmethod 
    def aggregate(X_hat, weights, indices, H, W, p):
        N, _, _, n = X_hat.size()
        X = (X_hat * weights).permute(0, 3, 1, 2).view(N, n, -1)
        weights = weights.view(N, 1, -1).expand(-1, n, -1)
        X_sum = torch.zeros(N, n, (H-p+1) * (W-p+1), dtype=X.dtype, device=X.device)
        weights_sum = torch.zeros_like(X_sum)
        
        for i in range(N):
            X_sum[i, :, :].index_add_(1, indices[i, :], X[i, :, :])
            weights_sum[i, :, :].index_add_(1, indices[i, :], weights[i, :, :])
 
        return F.fold(X_sum, (H, W), p) / F.fold(weights_sum, (H, W), p)
         
    def step1(self, input_y, sigma):
        _, _, H, W = input_y.size() 
        k, p, w, s = self.k1, self.p1, self.window, self.step
        y_mean = torch.mean(input_y, dim=1, keepdim=True) # for color
        indices = self.block_matching(y_mean, k, p, w, s)
        Y = self.gather_groups(input_y, indices, k, p)
        X_hat, weights = self.denoise1(Y, sigma)
        x_hat = self.aggregate(X_hat, weights, indices, H, W, p)
        return x_hat

    def step2(self, input_z, input_x, input_y, sigma, tau, indices=None):
        N, C, H, W = input_y.size() 
        k, p, w, s = self.k2, self.p2, self.window, self.step
        z_block = torch.mean(input_z, dim=1, keepdim=True) # for color
        if indices is None: indices = self.block_matching(z_block, k, p, w, s)
        X = self.gather_groups(input_x, indices, k, p)
        Y = self.gather_groups(input_y, indices, k, p)
        Z = self.gather_groups(input_z, indices, k, p)
        X_hat, Z_hat, weights = self.denoise2(Z, X, Y, sigma, tau)
        x_hat = self.aggregate(X_hat, weights, indices, H, W, p)
        z_hat = self.aggregate(Z_hat, weights, indices, H, W, p)
        return x_hat, z_hat, indices
    
    def forward(self, y, sigma=25.0, constraints='affine', method='n2n',
                p1=11, p2=6, k1=16, k2=64, w=65, s=3, M=9):
        self.set_parameters(sigma, constraints, method, p1, p2, k1, k2, w, s, M)
        z, x  = y, self.step1(y, sigma) # first pilot
        for m in range(self.M):
            tau = (1-(m+1)/self.M)*0.75
            if m%3==0:
                x, z, indices = self.step2(z, x, y, sigma, tau)
            else:
                x, z, indices = self.step2(z, x, y, sigma, tau, indices)
        return z
