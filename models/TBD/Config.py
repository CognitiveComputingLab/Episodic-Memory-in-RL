import torch.nn as nn

from dataclasses import dataclass
import inspect
from typing import Callable
from configs import EnvConfig
from models.DT import DT_Config
from models.DTi.Config import DTi_Config
from models.TBD.TBD import TBD

@dataclass
class TBD_Config(DTi_Config):

    lr_T: float
    lr_V: float
    lr_R: float

    def get_model(self, env_config: EnvConfig):
        from models.DT.embeddings import ActionEmbed, RTG_Embed, StateEmbed, TimeEmbed, ActionUnembed

        return TBD(
            context_len=self.context_len,
            embed_dim=self.embed_dim,
            num_persistent=self.num_persistent,
            act_dim=env_config.act_dim,
            obs_dim=env_config.obs_dim,
            max_action=env_config.max_action,
            max_ep_len=env_config.max_ep_len,
            is_discrete=env_config.is_discrete,
            time_embed=TimeEmbed(self, env_config),
            rtg_embed=RTG_Embed(env_config, self.embed_dim),
            state_embed=StateEmbed(env_config, self.embed_dim),
            action_embed=ActionEmbed(env_config, self.embed_dim),
            action_unembed=ActionUnembed(env_config, self.embed_dim),
            blocks=self.build_blocks(self.block_layout, self),
            lr_T=self.lr_T,
            lr_V=self.lr_V,
            lr_R=self.lr_R
        )