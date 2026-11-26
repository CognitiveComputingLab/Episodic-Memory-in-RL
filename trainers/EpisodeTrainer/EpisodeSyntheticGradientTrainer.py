import torch

from models.DTi import SG_Block
from trainers import BaseTrainer
from trainers.EpisodeTrainer import EpisodeTrainConfig, EpisodeTrainer
from trainers.TrainingContracts import StepSpec, TrainingBatch

from torch.nn.functional import mse_loss

class EpisodeSyntheticGradientTrainer(EpisodeTrainer):

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


        sg_modules = [module for module in self.model.modules() if isinstance(module, SG_Block)]

        for i, window in enumerate(windows):

            is_last_window = i == len(windows)-1

            self.model.detach_states()                     # truncation point (as before, incl. step 0)

            for module in sg_modules:
                module.track_state()
            
            preds = []
            for j,step in enumerate(window):
                is_last_step = j == len(window)-1

                sl = slice(step.start, step.end)
                preds.append(self.model(r[:, sl], s[:, sl], a[:, sl], t[:, sl]))

                if not is_last_step and not is_last_window:
                    for module in sg_modules:
                        module.track_state()

            lo, hi = window[0].start, window[-1].end
            m_w = m[:, lo:hi]
            if m_w.sum() == 0:                             # fully padded window: model.loss would be 0/0
                continue
            loss_w = self.model.loss(a[:, lo:hi], torch.cat(preds, dim=1), m_w) * (m_w.sum() / n_total)
            if not is_last_window:
                for module in sg_modules:
                    loss_w += module.future_loss()

            self.optimization.backward(loss_w)             # frees this window's graph
            total_loss += loss_w.detach()

            if not is_last_window:
                loss_f = None
                for module in sg_modules:
                    loss_f = loss_f + module.sg_loss(mse_loss) if loss_f else module.sg_loss(mse_loss)

                if loss_f:
                    total_loss += loss_f.detach()

    
        return self.optimization.apply(self.model.parameters(), total_loss)