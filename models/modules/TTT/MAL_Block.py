from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from models import Cacheful
from models.modules.TTT import MemoryFactory, titans_mlp
from models.modules.Transformer import Attention

class MAL_Block(nn.Module, Cacheful):
    """
    A Titans MAL block. This differs from the MAL block outlined in the paper as there is a residual connection.
    Also, the read comes before the attention and the write after. Reading before writing allows for the assoc scan of gradients
    from the write step to be ignored. If write comes before read, ignoring the assoc scan violates causality. Other potential
    benefits of reading first include allowing the model to choose what to remember based on what is already in memory. Writing
    has been placed after attention to allow the model to process what was read before then choosing what needs to be written.
    I am not sure if this has any consequences for when the model tries to then read that written information in the next pass -
    should be tested. This decision however does mean a residual stream is needed, otherwise only information from the past comes
    from the read. 
    """

    def __init__(
            self, 
            context_len: int, 
            e_dim: int,
            heads: int = 1, 
            num_persistent: int = 0,
            mini_batch_len: int = 1,
            mem_module: MemoryFactory = titans_mlp(ff_mult=4),
            ) -> None:
        super().__init__()

        self.attention = Attention(context_len + num_persistent, e_dim, heads)
        self.mem = mem_module(e_dim)

        self.norm1 = nn.RMSNorm(e_dim)
        self.norm2 = nn.RMSNorm(e_dim)

        self.e_dim = e_dim
        self.num_persistent = num_persistent
        self.persistent_tokens = nn.Parameter(torch.zeros([num_persistent, e_dim]))
        nn.init.trunc_normal_(self.persistent_tokens, std=0.02)     # Suggested by claude. Worth further investigation

        assert mini_batch_len <= context_len
        self.context_len = context_len
        self.mini_batch_len = mini_batch_len

        self.token_counter = self.context_len
        self.cache: Optional[Tensor] = None
        
    def forward(self, x: Tensor):
        b, t, _ = x.shape

        if self.token_counter == 0:
            x = torch.cat([self.persistent_tokens.expand([b,-1,-1]), x], dim=1)     # Maybe do this after mem read

        x = x + self.mem(self.norm1(x))         # Read mem. Could try cross attention here or concatination. Maybe don't pass in persistent
        x = x + self.attention(self.norm2(x))   # Self attention

        if self.token_counter == 0:
            x = x[:, self.num_persistent:, :]       # Trim out persistent tokens

        if not self.training and self.use_cache:
            assert self.cache is not None
            self.cache[:,self.token_counter:self.token_counter+t,:] = x     # grow cache
            to_store = self.cache
            self.token_counter += t
        else:
            to_store = x

        if self.token_counter == self.context_len:           # Only need to write at the end of a chunk as attention handles intra-chunk info
            for i in range(0, self.token_counter, self.mini_batch_len):
                self.mem.write(to_store[:, i:i+self.mini_batch_len, ...])
        
        return x

    def reset_cache(self, batch: int) -> None:
        device = self.norm1.weight.device
        self.token_counter = 0
        self.cache = torch.zeros([batch, self.context_len, self.e_dim], device=device)

    def wipe_cache(self) -> None:
        self.token_counter = self.context_len
        self.cache = None

    def enable_cache(self, use_cache: bool) -> None:
        super().enable_cache(use_cache)
        if not use_cache:
            self.token_counter = self.context_len
