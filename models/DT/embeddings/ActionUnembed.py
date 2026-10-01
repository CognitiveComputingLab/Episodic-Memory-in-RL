import torch.nn as nn
from torch import Tensor

from configs import EnvConfig

class ActionUnembed(nn.Module):
    def __init__(self, env_config: EnvConfig, e_dim: int) -> None:
        super().__init__()

        id = env_config.env_id
        
        if id == 'xmaze': 
            self.decoder = nn.Linear(e_dim, env_config.act_dim)
        elif id == 'mujoco':
            self.decoder = nn.Sequential(nn.Linear(e_dim, env_config.act_dim), nn.Tanh())
        else:
            raise ValueError("id '" + id + "' not recognised.")
    
    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)