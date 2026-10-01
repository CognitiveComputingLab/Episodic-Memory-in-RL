import torch

from trainers import BaseTrainer
from trainers.EpisodeTrainer import EpisodeTrainConfig

class EpisodeTrainer(BaseTrainer[EpisodeTrainConfig]):

    def __init__(self, train_config: EpisodeTrainConfig) -> None:
        super().__init__(train_config)
        self.detach_every = train_config.detach_every

    def iter(self) -> float:

        batch = next(self.iter_data)
        batch = tuple(x.to(self.device, non_blocking=True) for x in batch)
        self.model.reset_states(self.train_config.batch_size)
        a_preds = []

        s, a, r, t, m = batch

        for i in range(0, s.shape[1], self.model_config.context_len):
            if self.detach_every is not None and i % self.detach_every == 0:
                self.model.detach_states()
            sc, ac, rc, tc, mt = tuple(x[:, i:i + self.model_config.context_len] for x in batch)
            a_preds.append(self.model(rc, sc, ac, tc))

        a_pred = torch.cat(a_preds, dim=1)

        loss = self.model.loss(a, a_pred, m)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
        self.optim.step()

        return loss.item()
