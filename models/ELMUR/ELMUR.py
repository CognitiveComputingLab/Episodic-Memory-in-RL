from typing import List

from gymnasium import Env
import torch
from torch.nn.modules import Module, ModuleList

from models.DT import DecisionTransformer
from models.sampling import chunked_cacheful_sample

class ELMUR(DecisionTransformer):

    def __init__(
            self, 
            context_len: int, 
            embed_dim: int, 
            act_dim: int, 
            obs_dim: int, 
            max_action: float, 
            max_ep_len: int, 
            is_discrete: bool, 
            time_embed: Module, 
            rtg_embed: Module, 
            state_embed: Module, 
            action_embed: Module, 
            action_unembed: Module, 
            blocks: ModuleList,
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