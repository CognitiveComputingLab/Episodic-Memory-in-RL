from dataclasses import dataclass
from typing import Optional

from configs import TrainConfig
from trainers import BaseTrainer

@dataclass
class EpisodeTrainConfig(TrainConfig):

    detach_every: Optional[int]     # Detach the memory state every n chunks

    def get_trainer(self) -> BaseTrainer:
        from trainers.EpisodeTrainer import EpisodeTrainer
        return EpisodeTrainer(self)