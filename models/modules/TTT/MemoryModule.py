from abc import ABC, abstractmethod
from typing import Any

import torch

class MemoryModule(torch.nn.Module, ABC):

    @torch.compiler.disable
    @abstractmethod
    def write(self, x: Any) -> Any:
        pass