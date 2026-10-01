
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from torch.utils.data import IterableDataset

from datahandling.sources.DataSource import DataSource

@dataclass
class DatasetConfig(ABC):

    workers: int

    @abstractmethod
    def get_dataset(self, source: DataSource, reward_scale: int) -> IterableDataset:
        pass
    
    @abstractmethod
    def get_collate_fn(self) -> Callable:
        pass