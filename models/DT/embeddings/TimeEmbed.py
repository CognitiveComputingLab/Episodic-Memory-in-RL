import math

import torch
import torch.nn as nn
from torch import Tensor

from configs import EnvConfig
from models.DT import DT_Config

class TimeEmbed(nn.Module):
    def __init__(self, config: DT_Config, env_config: EnvConfig) -> None:
        super().__init__()

        if config.time_embed == 'global_sinusoidal':
            self.embedding = SinusoidalEmbedding(config.embed_dim, env_config.max_ep_len)
        elif config.time_embed == 'global_learnt':
            self.embedding = LearntEmbedding(config.embed_dim)
        elif config.time_embed == 'chunked_sinusoidal':
            embedding = SinusoidalEmbedding(config.embed_dim, config.context_len)
            self.embedding = ChunkedEmbedding(embedding, config.context_len)
        elif config.time_embed == 'chunked_learnt':
            embedding = LearntEmbedding(config.embed_dim)
            self.embedding = ChunkedEmbedding(embedding, config.context_len)
        else:
            raise ValueError(f"Unknown time_embed: {config.time_embed}")

    def forward(self, t: Tensor) -> Tensor:
        return self.embedding(t)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, e_dim: int, max_len: int = 5000):
        super().__init__()
        assert e_dim % 2 == 0, "SinusoidalEmbedding requires even e_dim"

        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)          # [max_len, 1]
        div_term = torch.exp(torch.arange(0, e_dim, 2) * (-math.log(10000.0) / e_dim))

        self.pe : Tensor
        pe = torch.zeros(max_len, e_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)                          # [max_len, e_dim]

    def forward(self, t: Tensor) -> Tensor:
        t = t.clamp(max=self.max_len-1)
        return self.pe[t]                                      # [B, T, e_dim]


class LearntEmbedding(nn.Module):
    def __init__(self, e_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, e_dim),
            nn.GELU(),
            nn.Linear(e_dim, e_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        t_norm = t.float().unsqueeze(-1) / t.shape[1]          # [B, T, 1], normalised 0→1
        return self.mlp(t_norm)                                 # [B, T, e_dim]


class ChunkedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Module, chunk_len: int):
        super().__init__()
        self.chunk_len = chunk_len
        self.embedding = embedding

    def forward(self, t: Tensor) -> Tensor:
        return self.embedding(t % self.chunk_len)              # [B, T, e_dim]