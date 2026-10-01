from gymnasium import Env
import torch
import torch.nn as nn

from models.DT.DT import DecisionTransformer
from models.sampling import chunked_cacheful_sample


class DecisionTitan(DecisionTransformer):

    def __init__(
            self, 
            context_len: int, 
            embed_dim: int, 
            titan_layers: list[int],    # On which layers to use TTT memory
            num_persistent: int,        # How many persistent tokens appended to the beginning of a chunk (unique for each layer)
            mini_batch_size: int,       # How often a write step occurs
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
            blocks: nn.ModuleList) -> None:
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

        self.titan_layers = titan_layers
        self.num_persistent = num_persistent
        self.mini_batch_size = mini_batch_size
        

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