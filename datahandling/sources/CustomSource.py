from typing import Callable, Optional

import numpy as np

from datahandling.sources.DataSource import EpisodeData, DataSource
from datahandling.utils import compute_obs_stats


class CustomSource(DataSource):
    """
    Wraps a list/dict-style offline dataset as an EpisodeSource.

    Expects each trajectory to be a dict with keys:
        "observations": np.ndarray  (T, obs_dim)  [no terminal obs]
        "actions":      np.ndarray  (T, act_dim)
        "returns":      np.ndarray  (T, 1)  (pre-computed RTG)

    Usage:
        source = CustomSource(get_dataset, discete_actions, obs_mean, obs_std, traj_lens)
    """

    def __init__(
        self,
        get_dataset: Callable[[], np.ndarray],
        discrete_actions: bool,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        traj_lens: np.ndarray,
    ):
        self._get_dataset = get_dataset
        self.discrete_actions = discrete_actions
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        self._episode_lens = traj_lens
        self.dataset: Optional[np.ndarray] = None


    def _ensure_loaded(self) -> None:
        if self.dataset is None:
            self.dataset = self._get_dataset()

    def __len__(self) -> int:
        return len(self._episode_lens)

    def get_episode(self, idx: int) -> EpisodeData:
        self._ensure_loaded()
        traj = self.dataset[idx]

        act_type = np.int32 if self.discrete_actions else np.float32
        actions = np.asarray(traj["actions"], act_type)

        return EpisodeData(
            observations =  np.asarray(traj["observations"], np.float32),
            actions =       actions,
            rewards =       np.asarray(traj["rewards"], np.float32),
        )

    def episode_lens(self) -> np.ndarray:
        return self._episode_lens

    def obs_mean(self) -> np.ndarray:
        return self._obs_mean

    def obs_std(self) -> np.ndarray:
        return self._obs_std