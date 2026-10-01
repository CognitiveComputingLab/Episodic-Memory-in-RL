import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

class CrossAttention(nn.Module):

    def __init__(
            self,
            e_dim: int,
            heads: int = 1,
            ) -> None:
        super().__init__()

        assert e_dim % heads == 0

        self.e_dim = e_dim
        self.h_dim = e_dim // heads
        self.heads = heads

        self.q_proj = nn.Linear(e_dim, e_dim)
        self.kv_proj = nn.Linear(e_dim, 2 * e_dim)
        self.out_proj = nn.Linear(e_dim, e_dim)
    
    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        B, T, _ = x_q.shape

        q = rearrange(self.q_proj(x_q), "b t (h e) -> b h t e", h=self.heads)
        k, v = rearrange(self.kv_proj(x_kv), "b t (p h e) -> p b h t e", h=self.heads, p=2)

        
        out = F.scaled_dot_product_attention(q, k, v)   # dense, no mask needed
        out = rearrange(out, "b h t e -> b t (h e)")
        return self.out_proj(out)