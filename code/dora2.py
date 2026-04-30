import torch
import torch.nn as nn
 
 
class DoRAPaper(nn.Module):
    def __init__(self, W, rank, p=0.0, alpha=1.0, eps=1e-8):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if W.ndim != 2:
            raise ValueError("W must be a 2D weight matrix")

        W = W.detach().clone()
        self.register_buffer("W0", W)
        self.out_dim, self.in_dim = W.shape
        self.rank = rank
        self.alpha = alpha
        self.eps = eps

        m = W.norm(dim=0, keepdim=True).clamp_min(eps)
        self.magnitude = nn.Parameter(m)
        self.B = nn.Parameter(torch.zeros(self.out_dim, rank))
        self.A = nn.Parameter(torch.empty(rank, self.in_dim))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.dropout = nn.Dropout(p)

    def effective_weight(self):
        delta = (self.alpha / self.rank) * (self.B @ self.A)
        V_prime = self.W0 + delta
        norm = V_prime.norm(dim=0, keepdim=True).clamp_min(self.eps)
        return self.magnitude * (V_prime / norm)

    def forward(self, x):
        if x.shape[-1] != self.in_dim:
            raise ValueError("Input feature dimension does not match W shape")
        return self.dropout(x) @ self.effective_weight().T