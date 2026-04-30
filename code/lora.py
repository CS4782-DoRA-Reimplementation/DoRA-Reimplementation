import torch
import torch.nn as nn

class LoRA(nn.Module):

    def __init__(self, W, rank, p = 0.0, alpha = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.register_buffer("W", W)

        self.rank = rank
        self.p = p
        self.alpha = alpha
        self.out_dim, self.in_dim = W.shape

        # B ∈ R^(d×r) zeros, A ∈ R^(r×k) Kaiming uniform — paper Section 4.1
        self.B = nn.Parameter(torch.zeros([self.out_dim, self.rank]))  # (output_dim, rank)
        self.A = nn.Parameter(torch.empty([self.rank, self.in_dim]))   # (rank, input_dim)
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

        self.lora_dropout = nn.Dropout(self.p)

    def forward(self, x):
        assert x.shape[-1] == self.in_dim, "x does not align with input dimension in LoRA"
        base = x @ self.W.T  # (batch, input_dim) @ (input_dim, output_dim) = (batch, output_dim)

        x_dropped = self.lora_dropout(x)  # (batch, input_dim)
        lora = (x_dropped @ self.A.T) @ self.B.T  # (batch, output_dim)
        lora_scaled = (self.alpha / self.rank) * lora  # (batch, output_dim)

        return base + lora_scaled  # (batch, output_dim)

