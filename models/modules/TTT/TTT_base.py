from abc import ABC, abstractmethod

import torch

class TTT_Module(torch.nn.Module, ABC):

    @abstractmethod
    def write(self, x: torch.Tensor):
        pass