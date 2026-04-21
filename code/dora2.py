import torch
import torch.nn as nn


class DoRAPaper(nn.Module):


    def __init__(self, W, rank, p=0.0, alpha=1.0, eps=1e-8):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if W.ndim != 2:
            raise ValueError("W must be a 2D weight matrix")

        frozen_weight = W.detach().clone()
        self.register_buffer("W", frozen_weight)

        self.rank = rank
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.out_dim, self.in_dim = frozen_weight.shape

        row_norm = frozen_weight.norm(dim=1).clamp_min(self.eps)
        base_direction = frozen_weight / row_norm.unsqueeze(1)

        self.register_buffer("base_direction", base_direction)
        self.magnitude = nn.Parameter(row_norm.clone())

        self.A = nn.Parameter(torch.zeros(self.out_dim, self.rank))
        self.B = nn.Parameter(torch.randn(self.rank, self.in_dim) * 0.01)

        self.dropout = nn.Dropout(self.p)

    def _adapted_direction(self):
        delta_direction = self.A @ self.B
        scaled_delta = (self.alpha / self.rank) * delta_direction

        adapted_direction = self.base_direction + scaled_delta
        adapted_norm = adapted_direction.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return adapted_direction, adapted_norm

    def effective_weight(self):
        adapted_direction, adapted_norm = self._adapted_direction()
        adapted_direction = adapted_direction / adapted_norm

        return self.magnitude.unsqueeze(1) * adapted_direction

    def forward(self, x):
        if x.shape[-1] != self.in_dim:
            raise ValueError("x does not align with input dimension in DoRAPaper")

        adapted_direction, adapted_norm = self._adapted_direction()
        magnitude_scale = self.magnitude / adapted_norm.squeeze(1)

        base_out = x @ self.base_direction.T
        dropped_x = self.dropout(x)
        delta_out = (dropped_x @ self.B.T) @ self.A.T
        delta_out = (self.alpha / self.rank) * delta_out

        return (base_out + delta_out) * magnitude_scale.unsqueeze(0)
