import torch

from trainers import BaseTrainer
from trainers.EpisodeTrainer import EpisodeTrainConfig
from trainers.TrainingContracts import StepSpec, TrainingBatch

class EpisodeTrainer(BaseTrainer[EpisodeTrainConfig]):

    def __init__(self, train_config: EpisodeTrainConfig) -> None:
        super().__init__(train_config)
        self.detach_every = train_config.detach_every

    def iter(self) -> float:

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
        self.model.reset_states(self.train_config.batch_size)
        a_preds = []

        s, a, r, t, m = (
            batch.states,
            batch.actions,
            batch.returns,
            batch.timesteps,
            batch.mask,
        )

        steps = [
            StepSpec(
                start=i,
                end=min(i + self.model_config.context_len, s.shape[1]),
                detach_state=(
                    self.detach_every is not None
                    and i % (self.detach_every * self.model_config.context_len) == 0
                ),
            )
            for i in range(0, s.shape[1], self.model_config.context_len)
        ]
        for step in steps:
            if step.detach_state:
                self.model.detach_states()
            sc, ac, rc, tc, mt = tuple(
                x[:, step.start:step.end] for x in (s, a, r, t, m)
            )
            a_preds.append(self.model(rc, sc, ac, tc))

        a_pred = torch.cat(a_preds, dim=1)

        loss = self.model.loss(a, a_pred, m)
        return self.optimization.step(
            {"primary": loss},
            self.model.parameters(),
        )
