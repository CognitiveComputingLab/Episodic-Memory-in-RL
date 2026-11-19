from typing import Any, Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from models import Cacheful
from models.modules.TTT import MemoryFactory, titans_mlp, MemoryModule
from models.modules.Transformer import Attention, grouped_causal

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

        self.attention = Attention(context_len + num_persistent, e_dim, heads, mask_fn=grouped_causal(persistent=num_persistent))
        self.mem: MemoryModule = mem_module(e_dim)

        self.norm1 = nn.RMSNorm(e_dim)
        self.norm2 = nn.RMSNorm(e_dim)

        self.e_dim = e_dim
        self.num_persistent = num_persistent
        self.persistent_tokens = nn.Parameter(torch.zeros([num_persistent, e_dim]))
        nn.init.trunc_normal_(self.persistent_tokens, std=0.02)     # Suggested by claude. Worth further investigation

        assert mini_batch_len <= context_len
        self.context_len = context_len
        self.mini_batch_len = mini_batch_len

        self.token_counter = 0
        self.cache: Optional[Tensor] = None

        self.kv_proj = nn.Linear(e_dim, 2 * e_dim)
        self.q_proj = nn.Linear(e_dim, e_dim)
        self.lr_proj: nn.Module = nn.Sequential(nn.Linear(e_dim, 1), nn.Sigmoid())
        

    def read(self, x: Tensor):
        q = self.q_proj(x)
        return self.mem(q) # * self.lr_proj(q)      Gate to encourage writing of useful tokens 

    def write(self, x: Tensor):
        kv = self.kv_proj(x)    # Get kv from x
        k, v = rearrange(kv, "b ... (p e) -> p b ... e", p=2)
        lr = self.lr_proj(x)
        self.mem.write(x,k,v,lr)
    
    def forward(self, x: Tensor):
        b, t, _ = x.shape

        x = x + self.read(self.norm1(x))         # Read mem. Could try cross attention here or concatination.

        # Add in persistent tokens
        if self.token_counter == 0:
            x = torch.cat([self.persistent_tokens.expand([b,-1,-1]), x], dim=1)

        x = x + self.attention(self.norm2(x))   # Self attention

        # Trim out persistent tokens
        if self.token_counter == 0:
            x = x[:, self.num_persistent:, :]

        if not self.training and self.use_cache:
            assert self.cache is not None
            self.cache[:,self.token_counter:self.token_counter+t,:] = x     # Grow write cache
            to_store = self.cache
        else:
            to_store = x

        # Write
        self.token_counter += t
        if self.token_counter == self.context_len:           # Only need to write at the end of a chunk as attention handles intra-chunk info
            for i in range(0, self.token_counter, self.mini_batch_len):
                self.write(to_store[:, i:i+self.mini_batch_len, ...])

        # Always reset token counter if cache not in use
        if not self.use_cache:
            self.token_counter = 0
        
        return x

    def reset_cache(self, batch: int) -> None:
        device = self.norm1.weight.device
        self.token_counter = 0
        self.cache = torch.zeros([batch, self.context_len, self.e_dim], device=device)

    def wipe_cache(self) -> None:
        self.token_counter = 0
        self.cache = None

    def enable_cache(self, use_cache: bool) -> None:
        super().enable_cache(use_cache)
