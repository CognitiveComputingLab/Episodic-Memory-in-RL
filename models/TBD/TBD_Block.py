import copy
from typing import Any, Optional

from einops import rearrange
import torch
from torch.func import functional_call
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models import Cacheful
from models.modules.TTT import MemoryFactory, titans_mlp, MemoryModule
from models.modules.Transformer import Attention, grouped_causal

class TBD_Block(nn.Module, Cacheful):

    def __init__(
            self, 
            context_len: int, 
            e_dim: int,
            m_dim: int,
            heads: int = 1, 
            num_persistent: int = 0,
            mini_batch_len: int = 1,
            mem_module: MemoryFactory = titans_mlp(ff_mult=4),
            ) -> None:
        super().__init__()

        self.attention = Attention(context_len + num_persistent, e_dim, heads, mask_fn=grouped_causal(persistent=num_persistent))
        self.mem: MemoryModule = mem_module(m_dim)

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

        self.k_proj = nn.Linear(e_dim, m_dim)
        self.q_proj = nn.Linear(e_dim, m_dim)
        self.v_proj = nn.Linear(e_dim, m_dim)
        self.v_target = copy.deepcopy(self.v_proj)
        self.v_inverse = nn.Linear(m_dim, e_dim)

        self.lr_proj: nn.Module = nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid())
        self.v_loss: Optional[Tensor]
        self.r_loss: Optional[Tensor]

    @staticmethod
    def call_frozen(module, *args, **kwargs):
        """Run module so gradients flow to inputs, but not to its parameters."""
        state = {k: v.detach() for k, v in module.named_parameters()}
        state.update(dict(module.named_buffers()))  # keep buffers as-is
        return functional_call(module, state, args, kwargs)

    def read(self, x: Tensor):
        q = self.q_proj(x)
        lr = self.lr_proj(q)    # Could get this from k(x^)
        v_hat = self.mem(q)     # mem already frozen as not part of module parameters
        x_w_hat = self.call_frozen(self.v_inverse, v_hat)
        return self.v_target(x_w_hat) * lr     # Gate to encourage writing of useful tokens 

    def write(self, x: Tensor):
        k = self.k_proj(x)    # Get kv from x
        v = self.v_proj(x)

        lr: Tensor = self.lr_proj(k)
        self.mem.write(x,k,v,lr)

        if self.training:
            
            lr_det = lr.detach()



            x_hat = self.v_inverse(v.detach())
            r_loss = (lr_det * F.mse_loss(x_hat, x.detach(), reduction='none').mean(-1, keepdim = True)).mean()   # experiment with using lr here
            self.r_loss = r_loss if not self.r_loss else self.r_loss + r_loss

            with torch.no_grad():
                v_target = self.v_target(x)
            v_loss = (lr_det * F.mse_loss(v, v_target.detach(), reduction='none').mean(-1, keepdim=True)).mean()
            self.v_loss = v_loss if not self.v_loss else self.v_loss + v_loss

        
    def forward(self, x: Tensor):
        b, t, _ = x.shape

        x = x + self.read(self.norm1(x))         # Read mem. Could try cross attention here or concatination.

        # Add in persistent tokens
        if self.token_counter == 0:
            x = torch.cat([self.persistent_tokens.expand([b,-1,-1]), x], dim=1)
            if self.training:
                self.v_loss = self.r_loss = None

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
            #self.v_proj, self.v_target = self.v_target, self.v_proj     # swap

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
