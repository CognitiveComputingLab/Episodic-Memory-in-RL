import torch.nn as nn
from torch import Tensor

from configs import EnvConfig

class ActionEmbed(nn.Module):
    def __init__(self, env_config: EnvConfig, e_dim: int) -> None:
        super().__init__()
        id = env_config.env_id
        
        if id == 'xmaze':
            self.embedding = nn.Sequential(nn.Embedding(env_config.act_dim, e_dim), nn.Tanh())
        elif id == 'mujoco':
            self.embedding = nn.Linear(env_config.act_dim, e_dim)
        else:
            raise ValueError("id '" + id + "' not recognised.")
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)