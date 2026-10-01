import torch.nn as nn
from torch import Tensor

from configs import EnvConfig

class StateEmbed(nn.Module):
    def __init__(self, env_config: EnvConfig, e_dim: int) -> None:
        super().__init__()

        embedding: nn.Module
        
        embedding = nn.Linear(env_config.obs_dim, e_dim)

        self.embedding = embedding
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)
