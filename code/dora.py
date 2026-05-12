import torch
import torch.nn as nn


class DoRA(nn.Module):

    def __init__(
        self,
        W: torch.Tensor,
        rank: int,
        p: float = 0.0,
        alpha: float = 1.0,
        eps: float = 1e-6,
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

        magnitude = W.float().norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        self.magnitude = nn.Parameter(magnitude)  # always float32

        self.B = nn.Parameter(torch.zeros(self.out_dim, rank))
        self.A = nn.Parameter(torch.empty(rank, self.in_dim))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def delta_weight(self) -> torch.Tensor:
        return self.scale * (self.B @ self.A)

    def _w0(self, ref: torch.Tensor) -> torch.Tensor:
        return self.W0.to(device=ref.device, dtype=ref.dtype)

    def _weight_norm(self, delta_weight: torch.Tensor) -> torch.Tensor:
        V_prime = self._w0(delta_weight) + delta_weight
        norm = V_prime.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)
        if self.detach_norm:
            norm = norm.detach()
        return norm

    def effective_weight(self) -> torch.Tensor:
        delta = self.delta_weight()
        W0 = self._w0(delta)
        norm = self._weight_norm(delta)
        return self.magnitude * ((W0 + delta) / norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError("Input feature dimension does not match W shape")

        W0 = self._w0(x)
        delta = self.delta_weight().to(x.dtype)
        norm = self._weight_norm(delta)
        magnitude_scale = (self.magnitude.to(x.dtype) / norm).squeeze(-1)

        base = x @ W0.T
        lora_update = ((self.dropout(x) @ self.A.to(x.dtype).T) @ self.B.to(x.dtype).T) * self.scale

        return base + (magnitude_scale - 1.0) * base + magnitude_scale * lora_update
