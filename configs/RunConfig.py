from dataclasses import dataclass

from configs.DatasetConfig import DatasetConfig
from configs.EnvConfig import EnvConfig
from configs.ModelConfig import ModelConfig
from configs.TrainConfig import TrainConfig

@dataclass
class RunConfig():

    model: ModelConfig
    environment: EnvConfig
    dataset: DatasetConfig
    trainer: TrainConfig

    @property
    def envionment(self) -> EnvConfig:
        """Deprecated compatibility alias. Use environment."""
        return self.environment
