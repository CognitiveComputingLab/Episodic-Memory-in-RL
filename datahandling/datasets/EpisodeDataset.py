import numpy as np
import torch
from torch.utils.data import IterableDataset

from datahandling.sources.DataSource import DataSource
from datahandling.utils import pad_along_axis

class EpisodeDataset(IterableDataset):
    """
    Samples full episodes from any EpisodeSource, weighted by episode length.
    Yields (states, actions, returns, mask) as numpy arrays.

    - states:  (T, obs_dim)  normalised
    - actions: (T, act_dim)
    - returns: (T, 1)        returns-to-go
    - mask:    (T,)          all ones (no padding at sample time; handled in collate)
    """

    def __init__(self, source: DataSource, reward_scale: float = 1.0):
        super().__init__()
        self.source = source
        self.reward_scale = reward_scale

        lens = source.episode_lens()
        self.episode_indices = np.arange(len(source))
        self.sample_prob = lens / lens.sum()

    def _process(self, idx: int):
        ep = self.source.get_episode(idx)

        obs = (ep.observations[:] - self.source.obs_mean()) / self.source.obs_std()   # z-normalisation
        rtg = np.flip(np.cumsum(np.flip(ep.rewards))).copy() * self.reward_scale        # reward scaling
        mask = np.ones(ep.actions.shape[0], dtype=np.float32)

        return obs, ep.actions, rtg, mask

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(
            seed=worker_info.id if worker_info is not None else 0
        )
        while True:
            idx = int(rng.choice(self.episode_indices, p=self.sample_prob))
            yield self._process(idx)

    @staticmethod
    def collate(batch):
        """Pads a batch of full episodes to the longest episode in the batch."""
        states, actions, returns, masks = zip(*batch)
        max_len = max(s.shape[0] for s in states)

        def pad(x, pad_value=0.0):
            return pad_along_axis(x, max_len, pad_value=pad_value)

        states      = torch.stack([torch.from_numpy(pad(s)) for s in states])
        actions     = torch.stack([torch.from_numpy(pad(a)) for a in actions])
        returns     = torch.stack([torch.from_numpy(pad(r.reshape(-1, 1), pad_value=float(r[-1]))) for r in returns])
        timesteps   = torch.arange(max_len, dtype=torch.long).unsqueeze(0).repeat(len(batch), 1)
        masks       = torch.stack([torch.from_numpy(pad(m)) for m in masks])

        return states, actions, returns, timesteps, masks