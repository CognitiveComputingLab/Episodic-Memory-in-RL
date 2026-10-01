from typing import Optional

import minari
import numpy as np


from datahandling.sources.DataSource import DataSource, EpisodeData
from datahandling.utils import compute_obs_stats


class MinariSource(DataSource):
    """
    Wraps a Minari dataset as an EpisodeSource.

    Usage:
        raw = minari.load_dataset("pen-human-v2")
        source = MinariSource(raw)
    """

    def __init__(
        self,
        minari_dataset,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
    ):
        self.ds = minari_dataset

        self._episode_lens = np.array(
            [int(ep.actions.shape[0]) for ep in self.ds.iterate_episodes()],
            dtype=np.int64,
        )
        assert self._episode_lens.size > 0, "Minari dataset appears empty"

        if obs_mean is not None and obs_std is not None:
            self._obs_mean = np.asarray(obs_mean)
            self._obs_std = np.asarray(obs_std)
        else:
            all_obs = [np.asarray(ep.observations) for ep in self.ds.iterate_episodes()]
            self._obs_mean, self._obs_std = compute_obs_stats(all_obs)

    def __len__(self) -> int:
        return len(self._episode_lens)

    def get_episode(self, idx: int) -> EpisodeData:
        ep = next(self.ds.iterate_episodes(episode_indices=[int(idx)]))
        return EpisodeData(
            observations =  np.asarray(ep.observations, dtype=np.float32),
            actions =       np.asarray(ep.actions, dtype=np.float32),
            rewards =       np.asarray(ep.rewards, dtype=np.float32),
        )

    def episode_lens(self) -> np.ndarray:
        return self._episode_lens

    def obs_mean(self) -> np.ndarray:
        return self._obs_mean

    def obs_std(self) -> np.ndarray:
        return self._obs_std