from typing import List, Tuple

import numpy as np

def pad_along_axis(array: np.ndarray, pad_to: int, axis: int = 0, pad_value: float = 0.0) -> np.ndarray:
    pad_size = [(0, 0)] * array.ndim
    pad_size[axis] = (0, pad_to - array.shape[axis])
    return np.pad(array, pad_size, mode="constant", constant_values=pad_value)


def compute_obs_stats(observations: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std across a list of observation arrays."""
    all_obs = np.concatenate(observations, axis=0)
    mean = all_obs.mean(axis=0)
    std  = all_obs.std(axis=0) + 1e-6
    return mean, std
