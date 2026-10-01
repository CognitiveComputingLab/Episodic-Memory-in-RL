from dataclasses import dataclass

from configs import DatasetConfig
from datahandling.datasets import SubsequenceDataset
from datahandling.sources.DataSource import DataSource

@dataclass
class SubsequenceDatasetConfig(DatasetConfig):

    sequence_len: int

    def get_dataset(self, source: DataSource, reward_scale: int):
        return SubsequenceDataset(source, self.sequence_len, reward_scale)
    
    def get_collate_fn(self):
        return SubsequenceDataset.collate