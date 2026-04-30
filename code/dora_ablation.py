import torch
import torch.nn as nn


class DoRAAblation(nn.Module):
    """
    Paper-style DoRA adapter with switches for the magnitude ablation.

    Variants:
    - full: train magnitude and low-rank direction update
    - frozen_magnitude: train only the low-rank direction update
    - magnitude_only: train only the magnitude vector
    """

    def __init__(
        self,
        W,
        rank,
        p=0.0,
        alpha=1.0,
        eps=1e-8,
        detach_norm=False,
        train_magnitude=True,
        train_direction=True,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if W.ndim != 2:
            raise ValueError("W must be a 2D weight matrix")

        frozen_weight = W.detach().clone()
        self.register_buffer("W", frozen_weight)
        self.out_dim, self.in_dim = frozen_weight.shape
        self.rank = rank
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.detach_norm = detach_norm

        row_norm = frozen_weight.norm(dim=1).clamp_min(eps)  # (output_dim,)
        base_direction = frozen_weight / row_norm.unsqueeze(1)  # (output_dim, input_dim)
        self.register_buffer("base_direction", base_direction)

        magnitude = row_norm.clone()  # (output_dim,)
        if train_magnitude:
            self.magnitude = nn.Parameter(magnitude)
        else:
            self.register_buffer("magnitude", magnitude)

        if train_direction:
            # Keep no-op adapter init: A zeros + random B => A @ B starts at 0.
            self.A = nn.Parameter(torch.zeros(self.out_dim, rank))
            self.B = nn.Parameter(torch.randn(rank, self.in_dim) * 0.01)
        else:
            self.register_buffer("A", torch.zeros(self.out_dim, rank))
            self.register_buffer("B", torch.zeros(rank, self.in_dim))

        self.dropout = nn.Dropout(p)

    def _adapted_direction(self):
        direction_update = (self.alpha / self.rank) * (self.A @ self.B)  # (output_dim, input_dim)
        adapted = self.base_direction + direction_update
        adapted_norm = adapted.norm(dim=1, keepdim=True).clamp_min(self.eps)  # (output_dim, 1)
        if self.detach_norm:
            adapted_norm = adapted_norm.detach()
        return adapted, adapted_norm

    def effective_weight(self):
        adapted, adapted_norm = self._adapted_direction()
        return self.magnitude.unsqueeze(1) * (adapted / adapted_norm)

    def forward(self, x):
        if x.shape[-1] != self.in_dim:
            raise ValueError("x does not align with input dimension in DoRAAblation")

        adapted, adapted_norm = self._adapted_direction()
        magnitude_scale = self.magnitude / adapted_norm.squeeze(1)  # (output_dim,)

        base_out = x @ self.base_direction.T  # (..., output_dim)
        x_dropped = self.dropout(x)  # (..., input_dim)
        delta_out = (x_dropped @ self.B.T) @ self.A.T  # (..., output_dim)
        delta_out = (self.alpha / self.rank) * delta_out

        return (base_out + delta_out) * magnitude_scale.unsqueeze(0)


class DoRAFullAblation(DoRAAblation):
    def __init__(self, W, rank, p=0.0, alpha=1.0):
        super().__init__(
            W=W,
            rank=rank,
            p=p,
            alpha=alpha,
            train_magnitude=True,
            train_direction=True,
        )


class DoRAFrozenMagnitude(DoRAAblation):
    def __init__(self, W, rank, p=0.0, alpha=1.0):
        super().__init__(
            W=W,
            rank=rank,
            p=p,
            alpha=alpha,
            train_magnitude=False,
            train_direction=True,
        )


class DoRAMagnitudeOnly(DoRAAblation):
    def __init__(self, W, rank, p=0.0, alpha=1.0):
        super().__init__(
            W=W,
            rank=rank,
            p=p,
            alpha=alpha,
            train_magnitude=True,
            train_direction=False,
        )
