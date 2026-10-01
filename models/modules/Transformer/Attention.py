import torch
import torch.nn as nn
from torch import Tensor, device
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from einops import rearrange

from models import Cacheful

class Attention(nn.Module, Cacheful):

    def __init__(
            self,
            w_size: int,
            e_dim: int,
            heads: int = 1,
            ) -> None:
        super().__init__()

        assert e_dim % heads == 0

        self.e_dim = e_dim
        self.h_dim = e_dim // heads
        self.heads = heads
        self.w_size = w_size

        self.qkv_proj = nn.Linear(e_dim, 3 * e_dim)
        self.out_proj = nn.Linear(e_dim, e_dim)

        self.k_cache: Tensor | None = None
        self.v_cache: Tensor | None = None
        self.cache_idx: int = 0

        self.block_mask: BlockMask = self._make_block_mask(
            self.w_size,
            self.w_size,
            0,
            device('cpu')
        )


    def _make_block_mask(self, q_len: int, kv_len: int, q_offset: int, device: device) -> BlockMask:
        def causal_mask(b, h, q_idx, kv_idx):
            q_global = q_idx + q_offset
            return (q_global + 1) // 3 >= (kv_idx + 1) // 3

        return create_block_mask(
            causal_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "b t (p h e) -> p b h t e", h=self.heads, p=3)

        # Make sure precomputed blockmask is on the correct device
        if self.block_mask.kv_num_blocks.device != x.device:
            self.block_mask = self.block_mask.to(x.device)

        block_mask = self.block_mask    # By default, assume training will full context - precomputed mask
        if not self.training and self.use_cache:
            assert self.k_cache is not None and self.v_cache is not None and self.cache_idx is not None
            self.k_cache[:, :, self.cache_idx:self.cache_idx + T, :] = k
            self.v_cache[:, :, self.cache_idx:self.cache_idx + T, :] = v
            q_offset = self.cache_idx
            self.cache_idx += T

            k = torch.narrow(self.k_cache, 2, 0, self.cache_idx)
            v = torch.narrow(self.v_cache, 2, 0, self.cache_idx)

            block_mask = self._make_block_mask(T, self.cache_idx, q_offset, x.device)     # Inference - new mask
        elif T < self.w_size:           # Training on last chunk without full context - new mask
            block_mask = self._make_block_mask(T, T, 0, x.device)
        
        out = flex_attention(q, k, v, block_mask=block_mask)
        out = rearrange(out, "b h t e -> b t (h e)")
        return self.out_proj(out)

    def reset_cache(self, batch: int) -> None:
        device = self.qkv_proj.weight.device
        self.k_cache = torch.zeros(batch, self.heads, self.w_size, self.h_dim, device=device)
        self.v_cache = torch.zeros(batch, self.heads, self.w_size, self.h_dim, device=device)
        self.cache_idx = 0

    def wipe_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.cache_idx = 0
        