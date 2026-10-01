from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeData:
    """Source-agnostic episode representation passed to samplers."""
    observations: np.ndarray  # (T+1, *obs_shape)
    actions: np.ndarray       # (T, *act_shape)
    rewards: np.ndarray       # (T,)


class DataSource(ABC):
    """
    Abstract base for any offline RL data source.
    Subclasses are responsible for loading data and exposing episodes
    as EpisodeData. Observation normalisation stats are computed here
    so samplers never need to know about the underlying data format.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of episodes in the dataset."""
        pass

    @abstractmethod
    def get_episode(self, idx: int) -> EpisodeData:
        """Return a single episode by index."""
        pass

    @abstractmethod
    def episode_lens(self) -> np.ndarray:
        """Return array of episode lengths (number of transitions)."""
        pass

    @abstractmethod
    def obs_mean(self) -> np.ndarray:
        """Per-feature observation mean for normalisation."""
        pass

    @abstractmethod
    def obs_std(self) -> np.ndarray:
        """Per-feature observation std for normalisation."""
        pass