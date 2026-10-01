from typing import Optional

from gymnasium import Env
import torch
import torch.nn as nn
from torch import Tensor

from models import Cacheful, Stateful
from models.DT import DecisionTransformer
from models.sampling import chunked_cacheful_sample
from models.modules.Transformer import CrossAttention

class RATE(DecisionTransformer, Cacheful, Stateful[Optional[Tensor]]):

    def __init__(
            self,
            context_len: int,
            embed_dim: int,
            act_dim: int,
            obs_dim: int,
            max_action: float,
            max_ep_len: int,
            is_discrete: bool,
            time_embed: nn.Module,
            rtg_embed: nn.Module,
            state_embed: nn.Module,
            action_embed: nn.Module,
            action_unembed: nn.Module,
            blocks: nn.ModuleList,
            num_memory_tokens: int,
            mem_heads: int
            ) -> None:
        super().__init__(
            context_len, 
            embed_dim, 
            act_dim, 
            obs_dim, 
            max_action, 
            max_ep_len, 
            is_discrete, 
            time_embed, 
            rtg_embed, 
            state_embed, 
            action_embed, 
            action_unembed,
            blocks)

        self.context_len_token = context_len * 3
        self.mrv = CrossAttention(embed_dim, mem_heads)      # Memory retention valve
        self.num_memory_tokens: int = num_memory_tokens
        self.mem_zero = nn.Parameter(torch.zeros([num_memory_tokens, embed_dim]))
        nn.init.trunc_normal_(self.mem_zero, std=1)
        self.mem_t = None

    def forward(self, r: Tensor, s: Tensor, a: Tensor, t: Tensor):

        x = self._embed_and_stack(r,s,a,t)
        l_sum = x.shape[1]

        # Append mems to beginning
        assert self.mem_t is not None
        if self.token_counter == 0:
            x = torch.cat([self.mem_t, x], dim=1)
            
        # Calculate state positions
        state_idx = 1 
        if not self.training and self.use_cache:
            state_idx = (1-self.token_counter)%3

        # Append mems to end
        self.token_counter += l_sum
        if self.token_counter == self.context_len_token:
            x = torch.cat([x, self.mem_t], dim=1)

        # Apply transformer
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        # Create new memories and remove end mems at end of chunk
        if self.token_counter == self.context_len_token:
            next_mem = x[:,-self.num_memory_tokens:,:]
            self.mem_t = self.mrv(self.mem_t, next_mem)
            x = x[:,:-self.num_memory_tokens,:]

        # Remove starting mem
        if self.token_counter == l_sum:
            x = x[:,self.num_memory_tokens:,:]

        # Reset
        if not self.use_cache:
            self.token_counter = 0

        return self.action_head(x[:, state_idx::3, :]) * self.max_action    # Predicts actions from the states

    def reset_state(self, batch: int) -> None:
        self.mem_t = self.mem_zero.expand(batch, *self.mem_zero.shape).clone()

    def reset_state_at(self, indices: list[int]) -> None:
        if self.mem_t is not None:
            self.mem_t[indices] = self.mem_zero.expand(len(indices), *self.mem_zero.shape).clone()

    def get_state(self) -> Optional[Tensor]:
        return self.mem_t

    def set_state(self, state: Optional[Tensor]) -> None:
        self.mem_t = state

    def detach_state(self) -> None:
        if self.mem_t is not None:
            self.mem_t = self.mem_t.detach()

    @torch.no_grad()    # This must be no_grad and not inference mode otherwise grad calc for memory breaks.
    def sample(self, env: Env, target: int) -> float:
        return chunked_cacheful_sample(
            self,
            self.max_ep_len,
            self.obs_dim,
            self.act_dim,
            self.is_discrete,
            env,
            target
        )