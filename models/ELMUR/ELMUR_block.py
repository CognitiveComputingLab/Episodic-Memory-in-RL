from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from einops import repeat

from models import Cacheful, Stateful
from models.modules.Transformer import Attention, CrossAttention

# Need to adjust for full overwrite on first iteration
class ELMUR_Block(nn.Module, Cacheful, Stateful[Tuple[Optional[Tensor], Tensor]]):

    def __init__(
            self, 
            w_size: int, 
            e_dim: int, 
            heads: int,
            mem_tokens: int,
            lr: float,
            ff_mult: int = 4
            ) -> None:
        super().__init__()

        self.w_size = w_size
        self.e_dim = e_dim

        self.attention = Attention(w_size, e_dim, heads)
        self.ffn = nn.Sequential(
            nn.Linear(e_dim, e_dim * ff_mult),
            nn.GELU(),
            nn.Linear(e_dim * ff_mult, e_dim),
        )
        self.norm1 = nn.RMSNorm(e_dim)
        self.norm2 = nn.RMSNorm(e_dim)
        self.norm3 = nn.RMSNorm(e_dim)

        self.mem_tokens = mem_tokens
        self.mem_zero = nn.Parameter(torch.zeros([mem_tokens, e_dim]))    # Can maybe make it's own dim
        nn.init.trunc_normal_(self.mem_zero, std=1)

        self.mem_t = None
        self.mem_fnn = nn.Sequential(
            nn.Linear(e_dim, e_dim * ff_mult),
            nn.GELU(),
            nn.Linear(e_dim * ff_mult, e_dim),
        )
        self.mem_norm1 = nn.RMSNorm(e_dim)
        self.mem_norm2 = nn.RMSNorm(e_dim)

        self.mem_to_tok = CrossAttention(e_dim, heads)
        self.tok_to_mem = CrossAttention(e_dim, heads)
        self.lr = lr

        # cache
        self.token_counter = 0
        self.cache: Optional[Tensor] = None

    def forward(self, x: Tensor):
        assert self.mem_t is not None

        x = self.norm1(x + self.attention(x))               # Self attention
        x = self.norm2(x + self.mem_to_tok(x, self.mem_t))  # Condition on memories via cross attention
        x = self.norm3(x + self.ffn(x))                     # FFN

        # Update cache
        new_tc = self.token_counter + x.shape[1]
        if self.use_cache:
            assert self.cache is not None
            self.cache[:, self.token_counter:new_tc, :] = x
            to_store = self.cache
        else:
            to_store = x
        self.token_counter = new_tc

        # Write mems at the end of a chunk
        if (self.token_counter == self.w_size):
            batch_range = torch.arange(to_store.shape[0], device=x.device)
            old_mem = self.mem_t[batch_range,self.lru].unsqueeze(1)

            new_mem: Tensor = self.mem_norm1(self.tok_to_mem(old_mem, to_store) + old_mem)
            new_mem = self.mem_norm2(self.mem_fnn(new_mem) + new_mem)

            new_mem = (self.lr * new_mem + (1 - self.lr) * old_mem)                 # [B, 1, e_dim]
            idx = repeat(self.lru, 'b -> b 1 e', e=self.e_dim)
            self.mem_t = self.mem_t.scatter(1, idx, new_mem)

            self.lru = (self.lru + 1) % self.mem_tokens

        if not self.use_cache:
            self.token_counter = 0

        return x

    def reset_cache(self, batch: int) -> None:
        self.token_counter = 0
        self.cache = torch.zeros((batch, self.w_size, self.e_dim), device= self.norm1.weight.device)

    def wipe_cache(self) -> None:
        self.cache = None
        self.token_counter = 0

    def reset_state(self, batch: int) -> None:
        self.mem_t = self.mem_zero.expand(batch, *self.mem_zero.shape).clone()
        self.lru = torch.zeros(batch, dtype=torch.long, requires_grad=False, device=self.mem_t.device)

    def reset_state_at(self, indices: list[int]) -> None:
        if self.mem_t is not None:
            self.mem_t[indices] = self.mem_zero.expand(len(indices), *self.mem_zero.shape).clone()
            self.lru[indices] = 0

    def get_state(self) -> Tuple[Optional[Tensor], Tensor]:
        return self.mem_t, self.lru

    def set_state(self, state: Tuple[Optional[Tensor], Tensor]) -> None:
        self.mem_t, self.lru = state

    def detach_state(self) -> None:
        if self.mem_t is not None:
            self.mem_t = self.mem_t.detach()
