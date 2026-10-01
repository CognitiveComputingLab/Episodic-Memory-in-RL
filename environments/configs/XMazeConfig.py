from dataclasses import dataclass
from functools import partial

from gymnasium.core import Env
import numpy as np

from configs import EnvConfig
from datahandling.sources import CustomSource
from datahandling.sources.DataSource import DataSource

from xmaze_jetcobblestone import XMazeEnv, load_dataset_from_disk

@dataclass
class XMazeConfig(EnvConfig):

    min_instructions: int
    max_instructions: int       # non inclusive
    possible_instructions: int
    min_corridor_len: int
    max_corridor_len: int
    episodes: int
    clip: int | None
    encoding: str
    instruction_every: int = 1

    @property
    def env_id(self) -> str:
        return "xmaze"

    @property
    def obs_dim(self) -> int:
        return self.possible_instructions + (self.max_instructions - 1)

    @property
    def act_dim(self) -> int:
        return self.possible_instructions + 1

    @property
    def is_discrete(self) -> bool:
        return True

    def get_env(self) -> Env:
        return XMazeEnv(
            turns_range=range(self.min_instructions, self.max_instructions),
            colour_count=self.possible_instructions,
            init_corridor_range=range(self.min_corridor_len, self.max_corridor_len),
            instruction_every= self.instruction_every,
            obs_mode=self.encoding
        )
    
    def get_datasource(self) -> DataSource:
        path = "./data/XMaze/" + XMazeEnv.getDatasetName(
            min_instructions=self.min_instructions,
            max_instructions=self.max_instructions,
            possible_instructions=self.possible_instructions,
            min_corridor_len=self.min_corridor_len,
            max_corridor_len=self.max_corridor_len,
            episodes=self.episodes,
            clip=self.clip,
            encoding=self.encoding
        )
     
        data = np.load(path, allow_pickle=True)
        ds_info = {
            "obs_mean": data["obs_mean"],
            "obs_std": data["obs_std"],
            "traj_lens": data["traj_lens"],
        }

        return CustomSource(partial(self.get_dataset, path), self.is_discrete, ds_info["obs_mean"], ds_info["obs_std"], ds_info["traj_lens"])

    @staticmethod
    def get_dataset(path: str) -> np.ndarray:
        data = np.load(path, allow_pickle=True, mmap_mode='r')
        return data["dataset"]