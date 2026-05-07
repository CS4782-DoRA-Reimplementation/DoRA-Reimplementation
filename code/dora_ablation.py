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

        frozen_weight = W.detach().clone().float()
        self.register_buffer("W", frozen_weight)
        self.out_dim, self.in_dim = frozen_weight.shape
        self.rank = rank
        self.p = p
        self.alpha = alpha
        self.scale = alpha / rank
        self.eps = eps
        self.detach_norm = detach_norm

        row_norm = frozen_weight.norm(p=2, dim=1).clamp_min(eps)  # (output_dim,)
        base_direction = frozen_weight / row_norm.unsqueeze(1)  # (output_dim, input_dim)
        self.register_buffer("base_direction", base_direction)

        magnitude = row_norm.clone()  # always float32 for stability
        if train_magnitude:
            self.magnitude = nn.Parameter(magnitude)
        else:
            self.register_buffer("magnitude", magnitude)

        if train_direction:
            # No-op adapter init: B zeros + kaiming A => B @ A starts at 0,
            # matching DoRA3 and avoiding initial fp16 overflow.
            self.B = nn.Parameter(torch.zeros(self.out_dim, rank))
            self.A = nn.Parameter(torch.empty(rank, self.in_dim))
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        else:
            self.register_buffer("B", torch.zeros(self.out_dim, rank))
            self.register_buffer("A", torch.zeros(rank, self.in_dim))

        self.dropout = nn.Dropout(p)

    def _adapted_norm_fp32(self, device):
        # Compute the per-row norm of (W0_dir + delta_dir) entirely in fp32
        A_f = self.A.float().to(device)
        B_f = self.B.float().to(device)
        base_dir = self.base_direction.to(device)
        direction_update = self.scale * (B_f @ A_f)
        adapted = base_dir + direction_update
        norm = adapted.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)
        if self.detach_norm:
            norm = norm.detach()
        return norm  # fp32, shape (out_dim, 1)

    def effective_weight(self):
        norm = self._adapted_norm_fp32(self.base_direction.device)
        A_f = self.A.float()
        B_f = self.B.float()
        adapted = self.base_direction + self.scale * (B_f @ A_f)
        return self.magnitude.unsqueeze(1) * (adapted / norm)

    def forward(self, x):
        if x.shape[-1] != self.in_dim:
            raise ValueError("x does not align with input dimension in DoRAAblation")

        # All norm/magnitude math in fp32 to avoid fp16 overflow → NaN.
        norm = self._adapted_norm_fp32(x.device)  # fp32, (out_dim, 1)
        magnitude_fp32 = self.magnitude.to(x.device).float()
        magnitude_scale_fp32 = magnitude_fp32 / norm.squeeze(1)  # fp32 (out_dim,)
        magnitude_scale = magnitude_scale_fp32.to(x.dtype)

        base_direction = self.base_direction.to(device=x.device, dtype=x.dtype)
        A = self.A.to(device=x.device, dtype=x.dtype)
        B = self.B.to(device=x.device, dtype=x.dtype)

        base_out = x @ base_direction.T  # (..., out_dim)
        x_dropped = self.dropout(x)
        delta_out = self.scale * (x_dropped @ A.T) @ B.T  # (..., out_dim)

        return (base_out + delta_out) * magnitude_scale


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
