
import torch

from trainers import BaseTrainer
from trainers.SubsequenceTrainer import SubsequenceTrainConfig
from trainers.TrainingContracts import TrainingBatch

class SubsequenceTrainer(BaseTrainer[SubsequenceTrainConfig]):

    def iter(self) -> float:
        self.model.reset_states(self.train_config.batch_size)
        batch = TrainingBatch.from_collated(next(self.iter_data))
        batch = TrainingBatch(
            *(
                tensor.to(self.device, non_blocking=True)
                for tensor in (
                    batch.states,
                    batch.actions,
                    batch.returns,
                    batch.timesteps,
                    batch.mask,
                )
            )
        )
        s, a, r, t, m = (
            batch.states,
            batch.actions,
            batch.returns,
            batch.timesteps,
            batch.mask,
        )

        a_pred = self.model(r, s, a, t)
        loss = self.model.loss(a, a_pred, m)
        return self.optimization.step(
            {"primary": loss},
            self.model.parameters(),
        )