
import torch

from trainers import BaseTrainer
from trainers.SubsequenceTrainer import SubsequenceTrainConfig

class SubsequenceTrainer(BaseTrainer[SubsequenceTrainConfig]):

    def iter(self) -> float:
        self.model.reset_states(self.train_config.batch_size)
        batch = next(self.iter_data)
        s, a, r, t, m = (x.to(self.device, non_blocking=True) for x in batch)

        a_pred = self.model(r, s, a, t)
        loss = self.model.loss(a, a_pred, m)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
        self.optim.step()
        return float(loss.item())