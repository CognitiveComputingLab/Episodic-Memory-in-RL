from abc import ABC, abstractmethod
from dataclasses import dataclass
import gymnasium as gym

from datahandling.sources.DataSource import DataSource

@dataclass
class EnvConfig(ABC):

    max_action: int
    max_ep_len: int
    reward_scaling: int

    @property
    @abstractmethod
    def env_id(self) -> str:
        pass

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def act_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        pass

    @abstractmethod
    def get_env(self) -> gym.Env:
        pass
    
    @abstractmethod
    def get_datasource(self) -> DataSource:
        pass