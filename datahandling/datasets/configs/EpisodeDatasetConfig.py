from dataclasses import dataclass


from configs import DatasetConfig
from datahandling.datasets import EpisodeDataset
from datahandling.sources.DataSource import DataSource

@dataclass
class EpisodeDatasetConfig(DatasetConfig):

    def get_dataset(self, source: DataSource, reward_scale: int):
        return EpisodeDataset(source, reward_scale)
    
    def get_collate_fn(self):
        return EpisodeDataset.collate