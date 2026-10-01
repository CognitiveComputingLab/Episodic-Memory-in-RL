from configs import TrainConfig
from trainers import BaseTrainer

class SubsequenceTrainConfig(TrainConfig):

    def get_trainer(self) -> BaseTrainer:
        from trainers.SubsequenceTrainer import SubsequenceTrainer
        return SubsequenceTrainer(self)