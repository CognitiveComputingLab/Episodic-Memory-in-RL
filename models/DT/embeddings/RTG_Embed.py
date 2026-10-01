import torch.nn as nn
from torch import Tensor

from configs import EnvConfig

class RTG_Embed(nn.Module):
    def __init__(self, env_config: EnvConfig, e_dim: int) -> None:
        super().__init__()

        embedding: nn.Module
        
        embedding = nn.Linear(1, e_dim)

        self.embedding = embedding
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)