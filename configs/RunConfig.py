from dataclasses import dataclass

from configs.DatasetConfig import DatasetConfig
from configs.EnvConfig import EnvConfig
from configs.ModelConfig import ModelConfig
from configs.TrainConfig import TrainConfig

@dataclass
class RunConfig():

    model: ModelConfig
    envionment: EnvConfig
    dataset: DatasetConfig
    trainer: TrainConfig
