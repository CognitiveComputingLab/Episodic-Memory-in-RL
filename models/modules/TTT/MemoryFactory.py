
from typing import Callable
import torch.nn as nn

from models.modules.TTT import MemoryModule
from .TTT_Memory import TTT_Memory
from .TitansMemory import TitansMemory


MemoryFactory = Callable[[int], MemoryModule]

def ttt_mlp(ff_mult: int) -> MemoryFactory:
    def get_module(e_dim: int):
        return TTT_Memory(
            e_dim,
            nn.Sequential(
                nn.Linear(e_dim, e_dim*ff_mult),
                nn.GELU(),
                nn.Linear(e_dim*ff_mult, e_dim),
        ))
    return get_module

def titans_mlp(ff_mult: int) -> MemoryFactory:
    def get_module(e_dim: int):
        return TitansMemory(
            e_dim,
            nn.Sequential(
                nn.Linear(e_dim, e_dim*ff_mult),
                nn.GELU(),
                nn.Linear(e_dim*ff_mult, e_dim),
        ))
    return get_module

