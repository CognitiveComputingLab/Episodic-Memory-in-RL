from abc import ABC, abstractmethod

import torch

class FlattenableState(ABC):
    
    @abstractmethod
    def flat_state(self) -> torch.Tensor:
        pass

    @abstractmethod
    def flat_size(self) -> int:
        pass

