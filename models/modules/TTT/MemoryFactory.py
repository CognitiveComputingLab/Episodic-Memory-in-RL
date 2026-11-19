
from typing import Callable
import torch.nn as nn

from models.modules.TTT import MemoryModule
from .TTT_Memory import TTT_Memory
from .TitansMemory import TitansMemory


MemoryFactory = Callable[[int], MemoryModule]

def ttt_mlp(ff_mult: int) -> MemoryFactory:
    def get_module(m_dim: int):
        return TTT_Memory(
            lr=1,
            memoryModule=nn.Sequential(
                nn.Linear(m_dim, m_dim*ff_mult),
                nn.GELU(),
                nn.Linear(m_dim*ff_mult, m_dim),
        ))
    return get_module

def titans_mlp(ff_mult: int) -> MemoryFactory:
    def get_module(m_dim: int):
        return TitansMemory(
            e_dim=m_dim,
            lr=1,
            memoryModule=nn.Sequential(
                nn.Linear(m_dim, m_dim*ff_mult),
                nn.GELU(),
                nn.Linear(m_dim*ff_mult, m_dim),
        ))
    return get_module

