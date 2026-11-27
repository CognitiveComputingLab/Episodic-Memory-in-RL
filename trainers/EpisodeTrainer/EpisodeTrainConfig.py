from dataclasses import dataclass
from typing import Optional

from configs import TrainConfig
from trainers import BaseTrainer

@dataclass
class EpisodeTrainConfig(TrainConfig):

    detach_every: Optional[int]     # Detach the memory state every n chunks
    synthetic_grads: bool

    def get_trainer(self) -> BaseTrainer:
        if self.synthetic_grads:
            from trainers.EpisodeTrainer import EpisodeSyntheticGradientTrainer
            return EpisodeSyntheticGradientTrainer(self)
        from trainers.EpisodeTrainer import EpisodeTrainer
        return EpisodeTrainer(self)