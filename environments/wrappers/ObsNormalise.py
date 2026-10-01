import numpy as np
import gymnasium as gym

class ObsNormaliseWrapper(gym.ObservationWrapper):
    def __init__(self, env, mean: np.ndarray, std: np.ndarray):
        super().__init__(env)
        self.mean = mean
        self.std = std

    def observation(self, observation):
        return (observation - self.mean) / self.std