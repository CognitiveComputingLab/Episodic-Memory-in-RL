from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from models import Cacheful

MaskFn = Callable[[Tensor, Tensor], Tensor]   # (q_idx, kv_idx) -> bool, True = may attend


def grouped_causal(group_size: int = 3, group_offset: int = 1, persistent: int = 0) -> MaskFn:
    """
    The first `persistent` tokens attend to each other fully and are visible to every later token.
    After that prefix, tokens attend to their own group and earlier ones, with groups
    counted from the end of the prefix.
    """
    def fn(q_idx: Tensor, k_idx: Tensor) -> Tensor:
        q_p = q_idx - persistent    # position relative to the end of the prefix
        k_p = k_idx - persistent
        grouped = (q_p + group_offset) // group_size >= (k_p + group_offset) // group_size

        q_is_persistent = q_idx < persistent
        k_is_persistent = k_idx < persistent
        return k_is_persistent | (~q_is_persistent & grouped)
    return fn


def build_mask(mask_fn: MaskFn, size: int) -> Tensor:
    idx = torch.arange(size)
    mask = mask_fn(idx[:, None], idx[None, :])        # broadcasts to [size, size]
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if mask.shape != (size, size):
        raise ValueError(f"mask must be [{size}, {size}], got {tuple(mask.shape)}")
    return mask


class Attention(nn.Module, Cacheful):

    def __init__(
            self,
            w_size: int,
            e_dim: int,
            heads: int = 1,
            mask_fn: MaskFn = grouped_causal(),
            ) -> None:
        super().__init__()
        assert e_dim % heads == 0

        self.e_dim, self.h_dim, self.heads, self.w_size = e_dim, e_dim // heads, heads, w_size

        self.qkv_proj = nn.Linear(e_dim, 3 * e_dim)
        self.out_proj = nn.Linear(e_dim, e_dim)

        self.k_cache: Tensor | None = None
        self.v_cache: Tensor | None = None
        self.cache_idx: int = 0

        self.mask: Tensor
        self.register_buffer("mask", build_mask(mask_fn, w_size), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "b t (p h e) -> p b h t e", h=self.heads, p=3)

        if not self.training and self.use_cache:
            assert self.k_cache is not None and self.v_cache is not None
            off = self.cache_idx
            self.k_cache[:, :, off:off + T] = k
            self.v_cache[:, :, off:off + T] = v
            self.cache_idx += T
            k = self.k_cache[:, :, :self.cache_idx]
            v = self.v_cache[:, :, :self.cache_idx]
            mask = self.mask[off:off + T, :off + T]
        else:
            mask = self.mask[:T, :T]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
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
        