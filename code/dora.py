import torch
import torch.nn as nn

class DoRA(nn.Module):

    def __init__(self, W, rank, p = 0.0, alpha = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.register_buffer("W", W)

        self.rank = rank
        self.p = p
        self.alpha = alpha
        self.out_dim, self.in_dim = W.shape

        # ---- DoRA decomposition ----
        # direction (row-normalized weight)
        row_norm = W.norm(dim=1) + 1e-8  # (output_dim,)
        W_norm = W / row_norm.unsqueeze(1)  # (output_dim, input_dim)
        self.register_buffer("W_dir", W_norm)

        # magnitude so W_dir * m == W per row at init (m = row L2 norm)
        self.magnitude = nn.Parameter(row_norm.squeeze().clone())  # (output_dim,)

        # One low-rank matrix initialized as zeros, the other initialized randomly
        self.A = nn.Parameter(torch.zeros([self.out_dim, self.rank])) # (output_dim, rank)
        self.B = nn.Parameter(torch.randn([self.rank, self.in_dim]) * 0.01) # (rank, input_dim)

        self.dropout = nn.Dropout(self.p)

    def forward(self, x):
        assert x.shape[-1] == self.in_dim, "x does not align with input dimension in DoRA"

        W_eff = self.W_dir * self.magnitude.unsqueeze(1) # (output_dim, input_dim)
        base = x @ W_eff.T # (batch, input_dim) @ (input_dim, output_dim) = (batch, output_dim)

        x_dropped = self.dropout(x) # (batch, input_dim)
        # x(AB)T = (xBT)AT
        lora = (x_dropped @ self.B.T) @ self.A.T  # (batch, output_dim)
        lora_scaled = (self.alpha / self.rank) * lora 

        return base + lora_scaled
