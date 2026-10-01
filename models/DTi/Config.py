import torch.nn as nn

from dataclasses import dataclass
import inspect
from typing import Callable
from configs import EnvConfig
from models.DT import DT_Config
from models.DTi.DTi import DecisionTitan

BlockBuilder = Callable[..., nn.ModuleList]
_LAYOUTS: dict[str, BlockBuilder] = {}

def register_layout(layout_id: str):
    def deco(fn: BlockBuilder) -> BlockBuilder:
        if layout_id in _LAYOUTS:
            raise ValueError(f"layout '{layout_id}' already registered")
        _LAYOUTS[layout_id] = fn
        return fn
    return deco

@dataclass
class DTi_Config(DT_Config):

    block_layout: str
    titan_layers: list[int]
    num_persistent: int
    mini_batch_size: int

    def get_model(self, env_config: EnvConfig):
        from models.DT.embeddings import ActionEmbed, RTG_Embed, StateEmbed, TimeEmbed, ActionUnembed

        return DecisionTitan(
            context_len=self.context_len,
            embed_dim=self.embed_dim,
            titan_layers=self.titan_layers,
            num_persistent=self.num_persistent,
            mini_batch_size=self.mini_batch_size,
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
            blocks=self.build_blocks(self.block_layout, self)
        )

    @staticmethod
    def build_blocks(layout_id: str, config: "DTi_Config") -> nn.ModuleList:
        if layout_id not in _LAYOUTS:
            raise ValueError(layout_id + " is not a registered layout.")
        fn = _LAYOUTS[layout_id]
        needed = inspect.signature(fn).parameters.keys()
        missing = needed - vars(config).keys()
        if missing:
            raise TypeError(f"layout '{layout_id}' needs {missing}, not found on {type(config).__name__}")
        return fn(**{name: getattr(config, name) for name in needed})