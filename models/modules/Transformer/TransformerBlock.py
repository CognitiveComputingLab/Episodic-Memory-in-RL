import torch.nn as nn
from torch import Tensor

from models.modules.Transformer import Attention

class TransformerBlock(nn.Module):

    def __init__(self, w_size: int, e_dim: int, heads: int, ff_mult: int = 4) -> None:
        super().__init__()

        self.attention = Attention(w_size, e_dim, heads)
        self.ffn = nn.Sequential(
            nn.Linear(e_dim, e_dim * ff_mult),
            nn.GELU(),
            nn.Linear(e_dim * ff_mult, e_dim),
        )
        self.norm1 = nn.RMSNorm(e_dim)
        self.norm2 = nn.RMSNorm(e_dim)

    def forward(self, x: Tensor):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
