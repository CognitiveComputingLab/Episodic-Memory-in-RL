from dataclasses import dataclass
from typing import List

from configs import EnvConfig
from models.modules.Transformer import TransformerBlock
from models.DT import DT_Config
from models.ELMUR.ELMUR import ELMUR
from models.ELMUR.ELMUR_block import ELMUR_Block

import torch.nn as nn


@dataclass
class ELMUR_Config(DT_Config):
    elmur_layers: List[int]
    num_memory_tokens: int
    lr: float

    def get_model(self, env_config: EnvConfig):
        from models.DT.embeddings import ActionEmbed, RTG_Embed, StateEmbed, TimeEmbed, ActionUnembed

        blocks = nn.ModuleList([
            TransformerBlock(
                self.context_len * 3,
                self.embed_dim,
                self.heads
                ) if i not in self.elmur_layers else
            ELMUR_Block(
                self.context_len * 3,
                self.embed_dim,
                self.heads,
                self.num_memory_tokens,
                self.lr
                )
            for i in range(self.layers)
        ])

        return ELMUR(
            context_len=self.context_len,
            embed_dim=self.embed_dim,
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
            blocks=blocks
        )