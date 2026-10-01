import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from datahandling.sources.DataSource import DataSource
from datahandling.utils import pad_along_axis


class SubsequenceDataset(IterableDataset):
    """
    Samples fixed-length windows from any EpisodeSource, weighted by episode length.
    Yields (states, actions, returns, timesteps, mask) as numpy arrays.

    - states:    (seq_len, obs_dim)  normalised, padded
    - actions:   (seq_len, act_dim)  padded with zeros
    - returns:   (seq_len, 1)        returns-to-go, padded with last RTG value
    - timesteps: (seq_len,)          absolute timestep indices, padded with increasing ints
    - mask:      (seq_len,)          1 for real steps, 0 for padding
    """

    def __init__(self, source: DataSource, seq_len: int = 20, reward_scale: float = 1.0):
        super().__init__()
        self.source = source
        self.seq_len = seq_len
        self.reward_scale = reward_scale

        lens = source.episode_lens()
        self.num_episodes = len(source)
        self.sample_prob = lens / lens.sum()

    def _process(self, idx: int, start: int):
        ep = self.source.get_episode(idx)

        rtg     = np.flip(np.cumsum(np.flip(ep.rewards))).copy() * self.reward_scale   # Reward scaling

        end = start + self.seq_len
        s_obs     = ep.observations[start:end]
        s_actions = ep.actions[start:end]
        s_rtg     = rtg[start:end]
        timesteps = np.arange(start, end, dtype=np.int64)

        s_obs = (s_obs - self.source.obs_mean()) / self.source.obs_std()        # z-normalisation

        real_len = s_actions.shape[0]
        mask = np.zeros(self.seq_len, dtype=np.float32)
        mask[:real_len] = 1.0

        if real_len < self.seq_len:
            s_obs     = pad_along_axis(s_obs,     self.seq_len)
            s_actions = pad_along_axis(s_actions, self.seq_len)
            s_rtg     = pad_along_axis(s_rtg.reshape(-1, 1), self.seq_len,
                                       pad_value=float(s_rtg[-1]) if real_len > 0 else 0.0).squeeze(-1)

        return s_obs, s_actions, s_rtg, timesteps, mask

    def __iter__(self):

        while True:
            idx = random.randint(0, self.num_episodes-1)
            ep_len = int(self.source.episode_lens()[idx])
            start  = random.randint(0, max(ep_len - 1, 0))
            yield self._process(idx, start)

    @staticmethod
    def collate(batch):
        """Stacks a batch of fixed-length sequence samples — no padding needed."""
        states    = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        actions   = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        returns   = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32).unsqueeze(-1)
        timesteps = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.long)
        masks     = torch.tensor(np.stack([b[4] for b in batch]), dtype=torch.float32)

        return states, actions, returns, timesteps, masks