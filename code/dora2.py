import torch
import torch.nn as nn
 
 
class DoRA(nn.Module):
    def __init__(self, W, rank, eps=1e-8, detach_norm=True):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if W.ndim != 2:
            raise ValueError("W must be a 2D weight matrix")
 
        W = W.detach().clone()
        self.register_buffer("W0", W)
        self.out_dim, self.in_dim = W.shape
        self.rank = rank
        self.eps = eps
        self.detach_norm = detach_norm
 
        m = W.norm(dim=0, keepdim=True).clamp_min(eps)
        self.magnitude = nn.Parameter(m)
        self.B = nn.Parameter(torch.zeros(self.out_dim, rank))
        self.A = nn.Parameter(torch.empty(rank, self.in_dim))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
 
    def effective_weight(self):
        V_prime = self.W0 + self.B @ self.A
        norm = V_prime.norm(dim=0, keepdim=True).clamp_min(self.eps)
        if self.detach_norm:
            norm = norm.detach()
        return self.magnitude * (V_prime / norm)
 
    def forward(self, x):
        if x.shape[-1] != self.in_dim:
            raise ValueError("Input feature dimension does not match W shape")
        return x @ self.effective_weight().T