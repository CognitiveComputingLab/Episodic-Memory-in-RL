from dataclasses import dataclass
from configs import EnvConfig, ModelConfig
from models.DT.DT import DecisionTransformer

import torch.nn as nn

from models.modules.Transformer import TransformerBlock

@dataclass
class DT_Config(ModelConfig):

    layers: int
    heads: int
    embed_dim: int
    time_embed: str

    def get_model(self, env_config: EnvConfig):
        from models.DT.embeddings import ActionEmbed, RTG_Embed, StateEmbed, TimeEmbed, ActionUnembed

        blocks = nn.ModuleList([
            TransformerBlock(self.context_len * 3, self.embed_dim, self.heads)
            for _ in range(self.layers)
        ])

        return DecisionTransformer(
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