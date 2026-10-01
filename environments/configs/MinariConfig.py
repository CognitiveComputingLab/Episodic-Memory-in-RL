from dataclasses import dataclass
import gymnasium as gym
import minari

from configs import EnvConfig
from datahandling.sources import MinariSource


@dataclass
class MinariConfig(EnvConfig):
    
    dataset_name: str

    def __post_init__(self):
        dataset = minari.load_dataset(self.dataset_name)
        
        self._obs_space = dataset.observation_space
        self._act_space = dataset.action_space

    def get_env(self) -> gym.Env:
        return minari.load_dataset(self.dataset_name, True).recover_environment()

    def get_datasource(self):
        return MinariSource(minari.load_dataset(self.dataset_name, True))

    @property
    def env_id(self) -> str:
        if 'mujoco' in self.dataset_name:
            return 'mujoco'
        raise ValueError('Env id not set for dataset', self.dataset_name)

    @property
    def obs_dim(self) -> int:
        return self._obs_space.shape[0]

    @property
    def act_dim(self) -> int:
        if self.is_discrete:
            return self._act_space.n
        return self._act_space.shape[0]

    @property
    def is_discrete(self) -> bool:
        return isinstance(self._act_space, gym.spaces.Discrete)