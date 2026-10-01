from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trainers import BaseTrainer

@dataclass
class TrainConfig(ABC):
    lr: float
    batches: int
    batch_size: int

    eval_seed: int
    eval_every: int
    eval_eps: int
    target_return: int

    grad_clip: float
    device: str

    do_wandb: bool
    project_name: str
    group_name: Optional[str]
    run_name: str

    checkpoint_every: int
    checkpoint_path: Optional[str]
    resume_from: Optional[str]

    @abstractmethod
    def get_trainer(self) -> "BaseTrainer":
        pass