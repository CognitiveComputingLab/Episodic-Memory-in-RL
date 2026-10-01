from abc import abstractmethod
from typing import Any

from gymnasium import Env
import torch
import torch.nn as nn
from torch import Tensor

from models import Cacheful, Stateful

class BaseModel(nn.Module):

    def __init__(self, context_len: int) -> None:
        super().__init__()
        self.context_len = context_len

    @abstractmethod
    @torch.no_grad()
    def sample(self, env: Env, target: int) -> float:
        pass
    
    @abstractmethod
    def loss(self, y: Tensor, pred_y: Tensor, mask: Tensor) -> Tensor:
        pass

    def reset_caches(self, batch: int) -> None:
        for module in self.modules():
            if isinstance(module, Cacheful):
                module.reset_cache(batch)

    def wipe_caches(self) -> None:
        for module in self.modules():
            if isinstance(module, Cacheful):
                module.wipe_cache()

    def enable_caches(self, use_cache: bool) -> None:
        for module in self.modules():
            if isinstance(module, Cacheful):
                module.enable_cache(use_cache)

    def reset_states(self, batch: int) -> None:
        for module in self.modules():
            if isinstance(module, Stateful):
                module.reset_state(batch)

    def reset_states_at(self, indices: list[int]) -> None:
        for module in self.modules():
            if isinstance(module, Stateful):
                module.reset_state_at(indices)

    def get_states(self) -> dict[str, Any]:
        return {
            name: module.get_state()
            for name, module in self.named_modules()
            if isinstance(module, Stateful)
        }

    def set_states(self, state: dict[str, Any]) -> None:
        stateful = {
            name: module for name, module in self.named_modules()
            if isinstance(module, Stateful)
        }
        missing = stateful.keys() - state.keys()
        extra = state.keys() - stateful.keys()
        if missing or extra:
            raise ValueError(
                f"State/model mismatch — missing: {missing}, unexpected: {extra}"
            )
        for name, module in stateful.items():
            module.set_state(state[name])

    def detach_states(self) -> None:
        for module in self.modules():
            if isinstance(module, Stateful):
                module.detach_state()
