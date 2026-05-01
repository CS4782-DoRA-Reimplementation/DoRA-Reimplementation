import torch
import torch.nn as nn


class DoRA3(nn.Module):

    def __init__(
        self,
        W: torch.Tensor,
        rank: int,
        p: float = 0.0,
        alpha: float = 1.0,
        eps: float = 1e-8,
        detach_norm: bool = True,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be positive")
        if W.ndim != 2:
            raise ValueError("W must be a 2D weight matrix")

        W = W.detach().clone()
        self.register_buffer("W0", W)

        self.out_dim, self.in_dim = W.shape
        self.rank = rank
        self.scale = alpha / rank
        self.eps = eps
        self.detach_norm = detach_norm
        self.dropout = nn.Dropout(p)

        magnitude = W.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        self.magnitude = nn.Parameter(magnitude)


        self.B = nn.Parameter(torch.zeros(self.out_dim, rank))
        self.A = nn.Parameter(torch.empty(rank, self.in_dim))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def delta_weight(self) -> torch.Tensor:
        return self.scale * (self.B @ self.A)

    def _weight_norm(self, delta_weight: torch.Tensor) -> torch.Tensor:
        V_prime = self.W0 + delta_weight
        norm = V_prime.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)
        if self.detach_norm:
            norm = norm.detach()
        return norm

    def effective_weight(self) -> torch.Tensor:
        delta = self.delta_weight()
        norm = self._weight_norm(delta)
        return self.magnitude * ((self.W0 + delta) / norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError("Input feature dimension does not match W shape")

        delta = self.delta_weight()
        norm = self._weight_norm(delta)
        magnitude_scale = (self.magnitude / norm).squeeze(-1)

        base = x @ self.W0.T
        lora_update = ((self.dropout(x) @ self.A.T) @ self.B.T) * self.scale

        return base + (magnitude_scale - 1.0) * base + magnitude_scale * lora_update


DoRAPaperFixed = DoRA3
DoRA = DoRA3
