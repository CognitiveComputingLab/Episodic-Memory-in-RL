from gymnasium import Env
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, int32, float32
from einops import rearrange

from models import BaseModel, Cacheful
from models.sampling import sliding_window_sample

class DecisionTransformer(BaseModel, Cacheful):
    """
    Decision Transformer - Lili Chen et al. (https://arxiv.org/abs/2106.01345) 
    """

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
            blocks: nn.ModuleList
            ) -> None:
        super().__init__(context_len)

        self.context_len = context_len
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.max_action = max_action
        self.max_ep_len = max_ep_len
        self.is_discrete = is_discrete

        self.time_embed = time_embed
        self.rtg_embed = rtg_embed
        self.state_embed = state_embed
        self.action_embed = action_embed
        self.action_head = action_unembed

        self.blocks = blocks

        self.final_norm = nn.RMSNorm(embed_dim)

        self.token_counter = 0      # cache


    def forward(self, r: Tensor, s: Tensor, a: Tensor, t: Tensor):

        x = self._embed_and_stack(r,s,a,t)  # Stack inputs
            
        for block in self.blocks:           # Apply transformer
            x = block(x)
        x = self.final_norm(x)

        # Calculate position of state tokens
        state_idx = 1 
        if not self.training and self.use_cache:
            state_idx = (1-self.token_counter)%3
            self.token_counter += x.shape[1]
        return self.action_head(x[:, state_idx::3, :]) * self.max_action    # Predicts actions from the states


    def _embed_and_stack(self, r: Tensor, s: Tensor, a: Tensor, t: Tensor) -> Tensor:
        """
        TO-DO: Fix sliding window as stack gets different sizes
        Handles the following cases:
        r s a, r s, a_{t-1} r s, a
        t t t, t t, t-1     t t, t
        0    , 0  , 2          , 2
        """

        if self.is_discrete:
            a = a.squeeze(-1)
        l_r, l_s, l_a = r.shape[1], s.shape[1], a.shape[1]

        phase = self.token_counter % 3  # Which token first? 0 = r, 1 = s, a = 2

        t_lst = [self.time_embed(t)]*3

        n_old = 3 - phase
        if n_old < 3 and n_old < l_r + l_s + l_a:
            t_lst[-n_old:] = [self.time_embed(t-1)] * n_old

        # 2 - 1
        # 1 - 2
        # 0 - 0

        r = self.rtg_embed(r) + t_lst[0] if l_r > 0 else None
        s = self.state_embed(s) + t_lst[1] if l_s > 0 else None
        a = self.action_embed(a) + t_lst[2] if l_a > 0 else None

        embeds = [r,s,a]
        for _ in range(phase):
            embeds = embeds[1:] + [embeds[0]]
        embeds = [embed for embed in embeds if embed is not None]
        x = torch.stack(embeds, dim=2)
        x = rearrange(x, 'b t n e -> b (t n) e')

        return x

    def reset_cache(self, batch: int) -> None:
        self.token_counter = 0

    def wipe_cache(self) -> None:
        self.token_counter = 0
    
    def loss(self, y: Tensor, pred_y: Tensor, mask: Tensor) -> Tensor:
        if self.is_discrete:
            loss = F.cross_entropy(
                pred_y.reshape(-1, pred_y.size(-1)),
                y.reshape(-1).long(),
                reduction='none',
            ).reshape(y.shape[:2])
        else:
            loss = F.mse_loss(pred_y, y, reduction='none').mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    @torch.compiler.disable
    @torch.no_grad()
    def sample(self, env: Env, target: int) -> float:
        return sliding_window_sample(
            self,
            self.max_ep_len,
            self.obs_dim,
            self.act_dim,
            self.is_discrete,
            env,
            target
        )