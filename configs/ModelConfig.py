from abc import abstractmethod, ABC
from dataclasses import dataclass

from configs import EnvConfig
from models import BaseModel

@dataclass
class ModelConfig(ABC):

    context_len: int

    @abstractmethod
    def get_model(self, env_config: EnvConfig) -> BaseModel:
        pass