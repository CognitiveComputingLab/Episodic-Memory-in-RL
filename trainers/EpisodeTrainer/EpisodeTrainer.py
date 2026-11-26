import torch

from trainers import BaseTrainer
from trainers.EpisodeTrainer import EpisodeTrainConfig
from trainers.TrainingContracts import StepSpec, TrainingBatch

class EpisodeTrainer(BaseTrainer[EpisodeTrainConfig]):

    def __init__(self, train_config: EpisodeTrainConfig) -> None:
        super().__init__(train_config)
        self.detach_every = train_config.detach_every

    def iter(self) -> float:
        s, a, r, t, m = tuple(t.to(self.device, non_blocking=True) for t in next(self.iter_data))
        

        self.model.reset_states(self.train_config.batch_size)
        

        C, T = self.model_config.context_len, s.shape[1]
        steps = [StepSpec(start=i, end=min(i + C, T)) for i in range(0, T, C)]
        per_window = self.detach_every or len(steps)       # None -> one window = full BPTT
        windows = [steps[i:i + per_window] for i in range(0, len(steps), per_window)]

        n_total = m.sum()
        total_loss = torch.zeros((), device=self.device)
        self.optimization.zero_grad()

        for window in windows:
            self.model.detach_states()                     # truncation point (as before, incl. step 0)
            preds = []
            for step in window:
                sl = slice(step.start, step.end)
                preds.append(self.model(r[:, sl], s[:, sl], a[:, sl], t[:, sl]))

            lo, hi = window[0].start, window[-1].end
            m_w = m[:, lo:hi]
            if m_w.sum() == 0:                             # fully padded window: model.loss would be 0/0
                continue
            loss_w = self.model.loss(a[:, lo:hi], torch.cat(preds, dim=1), m_w) * (m_w.sum() / n_total)

            self.optimization.backward(loss_w)             # frees this window's graph
            total_loss += loss_w.detach()

        return self.optimization.apply(self.model.parameters(), total_loss)